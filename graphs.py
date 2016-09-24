import math
import tensorflow as tf
from functools import partial
from tensorflow.contrib.layers.python.layers import batch_norm


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(shape, 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(shape), dtype="float32", name=name)


def param(shape, name, init):
    return tf.Variable(tf.ones(shape) * init, dtype="float32", name=name)


def deep_neural_network(input_tensor, layers, dropout=None):

    assert len(layers) >= 2

    act = None
    trans = input_tensor
    hs = []

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    for i, nodes in enumerate(layers[1:]):

        w = param_norm([layers[i], nodes], "W%d" % i)
        b = param_norm([nodes], "b%d" % i)

        act = tf.matmul(trans, w) + b

        if i + 1 < len(layers):
            trans = tf.nn.relu(act)
            if dropout:
                trans = tf.nn.dropout(trans, dropout, seed=1)

        if i + 2 < len(layers):
            hs.append(act)

    return act, hs


def logistic_regression(input_tensor, input_size, output_size):
    y, _ = deep_neural_network(input_tensor, [input_size, output_size])
    return y


def rnn_no_layers(input_tensor, size, steps, graph_type, input_model):

    # TODO this probably doesn't work with a bi-directional RNN

    rnn_out = 2 * size if graph_type.startswith('bi_') else size

    input_layer = input_tensor                                        # (batch, steps, input)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, input)
    input_layer = tf.reshape(input_layer, [-1, size])                 # (steps * batch, input)
    input_layer = input_model(input_layer)                            # (steps * batch, state)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, state)

    cell = rnn_cell(graph_type, size)                                 # (steps, batch, state)

    output_layer = rnn(graph_type, cell, input_layer)                 # (steps * 2, batch, state)
    output_layer = tf.reshape(output_layer, [-1, rnn_out])            # (steps * batch, hidden * 2)
    output_layer = tf.split(0, steps, output_layer)                   # (steps, batch, output)

    return tf.transpose(output_layer, [1, 0, 2])                      # (batch, steps, output)


def recurrent_neural_network(
        input_tensor,
        input_size,
        output_size,
        steps,
        state_size,
        graph_type,
        input_model=None,
        output_model=None):

    if input_model is None:
        input_model = partial(logistic_regression, input_size=input_size, output_size=state_size)

    rnn_out = 2 * state_size if graph_type.startswith('bi_') else state_size

    if output_model is None:
        output_model = partial(logistic_regression, input_size=rnn_out, output_size=output_size)

    input_layer = input_tensor                                        # (batch, steps, input)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, input)
    input_layer = tf.reshape(input_layer, [-1, input_size])           # (steps * batch, input)
    input_layer = input_model(input_layer)                            # (steps * batch, state)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, state)

    cell = rnn_cell(graph_type, state_size)                           # (steps, batch, state)

    output_layer = rnn(graph_type, cell, input_layer)                 # (steps * 2, batch, state)
    output_layer = tf.reshape(output_layer, [-1, rnn_out])            # (steps * batch, hidden * 2)
    output_layer = output_model(output_layer)                         # (steps * batch, output)
    output_layer = tf.split(0, steps, output_layer)                   # (steps, batch, output)

    return tf.transpose(output_layer, [1, 0, 2])                      # (batch, steps, output)


def rnn_cell(graph_type, size):
    if graph_type.endswith('lstm'):
        return tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
    elif graph_type.endswith('gru'):
        return tf.nn.rnn_cell.GRUCell(size)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(size)


def rnn(graph_type, cell, x):
    if graph_type.startswith('bi_'):
        output, _, _ = tf.nn.bidirectional_rnn(cell, cell, x, dtype=tf.float32)
    else:
        output, _ = tf.nn.rnn(cell, x, dtype=tf.float32)
    return output


def ladder_network(x, layers, noise, denoising_cost=[math.pow(10, 2-k) for k in range(20)]):

    # TODO: Implement batch normalisation by hand

    h_clean = x
    h_corr = x

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    z_corrs = [h_corr]
    z_cleans = [h_clean]

    # Encoders
    for i, nodes in enumerate(layers[1:]):

        w = param_norm([layers[i], nodes], "W%d" % i)
        b = param_norm([nodes], "b%d" % i)

        z_clean = batch_norm(tf.matmul(h_clean, w) + b)
        z_corr = add_noise(batch_norm(tf.matmul(h_corr, w) + b), noise)

        z_cleans.append(z_clean)
        z_corrs.append(z_corr)

        h_clean = tf.nn.relu(z_clean)
        h_corr = tf.nn.relu(z_corr)

    z_est = h_corr
    reverse_layers = layers[::-1]
    d_cost = []

    # Decoder
    for j, nodes in enumerate(reverse_layers):

        i = len(layers) - (j + 1)

        if j != 0:
            v = param_norm([layers[i+1], nodes], "V%d" % i)
            c = param_norm([nodes], "c%d" % i)
            z_est = tf.matmul(z_est, v) + c

        z_est = batch_norm(z_est)
        z_corr = z_corrs[i]
        z_clean = z_cleans[i]

        z_est = combinator(z_est, z_corr, nodes)

        # z_est_bn = (z_est - mean) / var TODO ???
        z_est_bn = z_est

        d_cost.append(
            (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z_clean), 1)) / nodes) * denoising_cost[i])

    y_clean = h_clean
    y_corr = h_corr
    u_cost = tf.add_n(d_cost)

    return y_clean, y_corr, u_cost


def add_noise(x, noise_var):
    return x + tf.random_normal(tf.shape(x)) * noise_var


def combinator(z_est, z_corr, size):

    a1 = param([size], 'a1', 0.)
    a2 = param([size], 'a2', 1.)
    a3 = param([size], 'a3', 0.)
    a4 = param([size], 'a4', 0.)
    a5 = param([size], 'a5', 0.)
    a6 = param([size], 'a6', 0.)
    a7 = param([size], 'a7', 1.)
    a8 = param([size], 'a8', 0.)
    a9 = param([size], 'a9', 0.)
    a10 = param([size], 'a10', 0.)

    mu = tf.mul(a1, tf.sigmoid(tf.mul(a2, z_corr) + a3)) + tf.mul(a4, z_corr) + a5
    va = tf.mul(a6, tf.sigmoid(tf.mul(a7, z_corr) + a8)) + tf.mul(a9, z_corr) + a10

    return (z_est - mu) * va + mu

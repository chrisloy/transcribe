import numpy as np
import tensorflow as tf
from functools import partial
from tensorflow.python import control_flow_ops
from tensorflow.python.framework.ops import op_scope


eps = 1e-3


def flatten(l):
    return sum(map(lambda i: flatten(i) if type(i) == list else [i], l), [])


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(flatten(shape), 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(flatten(shape)), dtype="float32", name=name)


def param(shape, init, name=None, trainable=True):
    return tf.Variable(tf.ones(flatten(shape)) * init, dtype="float32", name=name, trainable=trainable)


def batch_norm_wrapper(z, is_training, decay=0.999):

    gamma = param([z.get_shape()[-1]], 1)
    beta = param([z.get_shape()[-1]], 0)
    pop_mean = param([z.get_shape()[-1]], 0, trainable=False)
    pop_var = param([z.get_shape()[-1]], 1, trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(z, [0])
        train_mean = tf.assign(pop_mean, tf.mul(pop_mean, decay) + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, tf.mul(pop_var, decay) + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(z, batch_mean, batch_var, beta, gamma, eps)
    else:
        return tf.nn.batch_normalization(z, pop_mean, pop_var, beta, gamma, eps)


def deep_neural_network(input_tensor, layers, dropout=None, batch_norm=False, training=None, init='gaussian'):

    assert len(layers) >= 2

    act = None
    trans = input_tensor

    hs = []

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    for i, nodes in enumerate(layers[1:]):

        if init == 'gaussian':
            w = param_norm([layers[i], nodes], "W%d" % i)
            b = param_norm([nodes], "b%d" % i)
        elif init == 'identity':
            w = tf.Variable(initial_value=np.eye(layers[i], nodes), name=("W%d" % i), dtype="float32")
            b = param_zeros([nodes], "b%d" % i)
        else:
            assert False, "Unexpected init command [%s]" % init

        act = tf.matmul(trans, w) + b
        hs.append(act)

        if batch_norm:
            act = control_flow_ops.cond(
                training,
                lambda: batch_norm_wrapper(act, True),
                lambda: batch_norm_wrapper(act, False)
            )

        if i + 1 < len(layers):
            trans = tf.nn.relu(act)
            if dropout:
                trans = tf.nn.dropout(trans, dropout, seed=1)
    return act, hs


def logistic_regression(input_tensor, input_size, output_size):
    return deep_neural_network(input_tensor, [input_size, output_size])[0]


def rnn_no_layers(input_tensor, size, steps, graph_type):

    # TODO Bi-directional RNN not currently supported

    if graph_type.startswith('bi_'):
        assert False, "Bi-directional RNNs not supported"

    input_layer = input_tensor                                        # (batch, steps, input)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, input)
    input_layer = tf.reshape(input_layer, [-1, size])                 # (steps * batch, input)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, state)

    cell = rnn_cell(graph_type, size)                                 # (steps, batch, state)

    output_layer = rnn(graph_type, cell, input_layer)                 # (steps * 2, batch, state)
    output_layer = tf.reshape(output_layer, [-1, size])               # (steps * batch, hidden * 2)
    output_layer = tf.split(0, steps, output_layer)                   # (steps, batch, output)

    return tf.transpose(output_layer, [1, 0, 2])                      # (batch, steps, output)


def recurrent_neural_network(
        input_tensor,
        state_size,
        output_size,
        steps,
        graph_type):

    n = 2 if graph_type.startswith('bi_') else 1
    rnn_out = n * state_size

    output_model = partial(logistic_regression, input_size=rnn_out, output_size=output_size)

    input_layer = input_tensor                                        # (batch, steps, state)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, state)
    input_layer = tf.reshape(input_layer, [-1, state_size])           # (steps * batch, state)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, state)

    cell = rnn_cell(graph_type, state_size)                           # (steps, batch, state)

    output_layer = rnn(graph_type, cell, input_layer)                 # (steps * n, batch, state)
    output_layer = tf.reshape(output_layer, [-1, rnn_out])            # (steps * batch, hidden * n)
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


def ladder_network(x, layers, noise, training, denoising_cost):

    def batch_norm(z, batch_mean, batch_var, gamma, beta, include_noise):
        with op_scope([z, batch_mean, batch_var, gamma, beta], None, "batchnorm"):
            z_out = (z - batch_mean) / tf.sqrt(tf.add(batch_var, eps))
            if include_noise:
                z_out = add_noise(z_out, noise)
            z_fixed = tf.mul(gamma, z_out) + beta
            return z_fixed, z_out

    def batch_norm_and_noise(z, is_training, include_noise, decay=0.99999):

        gamma = param([z.get_shape()[-1]], 1)
        beta = param([z.get_shape()[-1]], 0)
        pop_mean = param([z.get_shape()[-1]], 0, trainable=False)
        pop_var = param([z.get_shape()[-1]], 1, trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(z, [0])
            train_mean = tf.assign(pop_mean, tf.mul(pop_mean, decay) + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, tf.mul(pop_var, decay) + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return batch_norm(z, batch_mean, batch_var, gamma, beta, include_noise)
        else:
            return batch_norm(z, pop_mean, pop_var, gamma, beta, include_noise)

    h_clean = x
    h_corr = x

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    z_corrs = [h_corr]
    z_cleans = [h_clean]

    # Encoders
    for i, nodes in enumerate(layers[1:]):

        w = param_norm([layers[i], nodes], "W%d" % i)
        z_clean = tf.matmul(h_clean, w)
        z_corr = tf.matmul(h_corr, w)

        h_clean, z_clean = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_clean, True, False),
            lambda: batch_norm_and_noise(z_clean, False, False)
        )

        h_corr, z_corr = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_corr, True, True),
            lambda: batch_norm_and_noise(z_corr, False, True)
        )

        z_cleans.append(z_clean)
        z_corrs.append(z_corr)

        if i + 2 < len(layers):
            h_clean = tf.nn.relu(h_clean)
            h_corr = tf.nn.relu(h_corr)

    z_dec = h_corr
    reverse_layers = layers[::-1]
    dec_cost = []

    # Decoder
    for j, nodes in enumerate(reverse_layers):

        i = len(layers) - (j + 1)

        if j != 0:
            v = param_norm([layers[i+1], nodes], "V%d" % i)
            z_dec = tf.matmul(z_dec, v)

        _, z_dec = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_dec, True, False),
            lambda: batch_norm_and_noise(z_dec, False, False)
        )

        z_corr = z_corrs[i]
        z_clean = z_cleans[i]

        z_dec = combinator(z_dec, z_corr, nodes)

        cost = tf.reduce_mean(tf.reduce_sum(tf.square(z_dec - z_clean), 1)) / nodes
        dec_cost.append((cost * denoising_cost[i]))

    y_clean = h_clean
    y_corr = h_corr
    u_cost = tf.add_n(dec_cost)

    return y_clean, y_corr, u_cost


def add_noise(x, noise_var):
    return x + tf.random_normal(tf.shape(x)) * noise_var


def combinator(z_est, z_corr, size):

    a1 = param([size], 0., name='a1')
    a2 = param([size], 1., name='a2')
    a3 = param([size], 0., name='a3')
    a4 = param([size], 0., name='a4')
    a5 = param([size], 0., name='a5')
    a6 = param([size], 0., name='a6')
    a7 = param([size], 1., name='a7')
    a8 = param([size], 0., name='a8')
    a9 = param([size], 0., name='a9')
    a10 = param([size], 0., name='a10')

    mu = tf.mul(a1, tf.sigmoid(tf.mul(a2, z_corr) + a3)) + tf.mul(a4, z_corr) + a5
    va = tf.mul(a6, tf.sigmoid(tf.mul(a7, z_corr) + a8)) + tf.mul(a9, z_corr) + a10

    return (z_est - mu) * va + mu

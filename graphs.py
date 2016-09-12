import tensorflow as tf
from functools import partial


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(shape, 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(shape), dtype="float32", name=name)


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

    if output_model is None:
        output_model = partial(logistic_regression, input_size=state_size, output_size=output_size)

    initial_state = rnn_initial_state(graph_type, state_size)

    input_layer = input_tensor                                        # (batch, steps, input)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, input)
    input_layer = tf.reshape(input_layer, [-1, input_size])           # (steps * batch, input)
    input_layer = input_model(input_layer)                            # (steps * batch, state)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, state)

    cell = rnn_cell(graph_type, state_size)                           # (steps, batch, state)

    output_layer = rnn(graph_type, cell, initial_state, input_layer)  # (steps, batch, state)
    output_layer = tf.reshape(output_layer, [-1, state_size])         # (steps * batch, hidden)
    output_layer = output_model(output_layer)                         # (steps * batch, output)
    output_layer = tf.split(0, steps, output_layer)                   # (steps, batch, output)

    return tf.transpose(output_layer, [1, 0, 2]), initial_state       # (batch, steps, output)


def rnn_initial_state(graph_type, hidden):
    if graph_type == 'lstm' or graph_type == 'bi_lstm':
        return tf.placeholder(tf.float32, shape=[None, hidden * 2], name="initial_state")
    else:
        return tf.placeholder(tf.float32, shape=[None, hidden], name="initial_state")


def rnn_cell(graph_type, size):
    if graph_type == 'lstm' or graph_type == 'bi_lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
    elif graph_type == 'gru' or 'bi_gru':
        return tf.nn.rnn_cell.GRUCell(size)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(size)


def rnn(graph_type, cell, initial_state, x):
    if graph_type == 'bi_rnn' or graph_type == 'bi_lstm' or graph_type == 'bi_gru':
        outputs, _, _ = tf.nn.bidirectional_rnn(cell, cell, x, initial_state, initial_state)
        fw, bw = tf.split(2, 2, outputs)
        output = fw
    else:
        output, _ = tf.nn.rnn(cell, x, initial_state)
    return output

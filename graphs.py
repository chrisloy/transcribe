import tensorflow as tf


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(shape, 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(shape), dtype="float32", name=name)


def deep_neural_network(input_tensor, layers, dropout=None):

    assert len(layers) >= 2

    act = None
    trans = input_tensor

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    for i, nodes in enumerate(layers[1:]):

        w = param_norm([layers[i], nodes], "W%d" % i)
        b = param_norm([nodes], "b%d" % i)

        act = tf.matmul(trans, w) + b

        if i + 1 < len(layers):
            trans = tf.nn.relu(act)
            if dropout:
                trans = tf.nn.dropout(trans, dropout, seed=1)

    return act


def recurrent_neural_network(
        input_tensor,
        input_size,
        output_size,
        steps,
        hidden,
        graph_type):

    initial_state = rnn_initial_state(graph_type, hidden)

    w_h = param_norm([input_size, hidden], "w_h")
    b_o = param_norm([output_size], "b_o")
    w_o = param_norm([hidden, output_size], "w_o")
    b_h = param_norm([hidden], "b_h")

    input_layer = input_tensor                                        # (batch, steps, input)
    input_layer = tf.transpose(input_layer, [1, 0, 2])                # (steps, batch, input)
    input_layer = tf.reshape(input_layer, [-1, input_size])           # (steps * batch, input)
    input_layer = tf.matmul(input_layer, w_h) + b_h                   # (steps * batch, hidden)
    input_layer = tf.split(0, steps, input_layer)                     # (steps, batch, hidden)

    cell = rnn_cell(graph_type, hidden)                               # (steps, batch, hidden)

    output_layer = rnn(graph_type, cell, initial_state, input_layer)  # (steps, batch, hidden)
    output_layer = tf.reshape(output_layer, [-1, hidden])             # (steps * batch, hidden)
    output_layer = tf.matmul(output_layer, w_o) + b_o                 # (steps * batch, output)
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

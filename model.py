import sys
import tensorflow as tf


class Model:
    def __init__(self, x, y, y_gold, loss, train_step, i_state=None):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step
        self.report_name = "ERROR"
        self.report_target = loss
        self.i_state = i_state

    def set_report(self, name, target):
        self.report_name = name
        self.report_target = target

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_gold, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(shape, 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(shape), dtype="float32", name=name)


def feed_forward_model(
        features,
        output,
        learning_rate=0.001,
        hidden_nodes=list(),
        dropout=False,
        one_hot=False,
        with_bias=True):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    previous_nodes = features
    act = None
    trans = x

    depth = 0

    sys.stdout.write("Graph shape: %d" % previous_nodes)

    layers = hidden_nodes + [output]

    for i, nodes in enumerate(layers):

        sys.stdout.write(" --> %d" % nodes)
        depth += 1
        w = param_norm([previous_nodes, nodes], "W%d" % depth)

        if with_bias:
            b = param_norm([nodes], "b%d" % depth)
            act = tf.matmul(trans, w) + b
        else:
            act = tf.matmul(trans, w)

        if i + 1 < len(layers):
            trans = tf.nn.relu(act)
            if dropout:
                trans = tf.nn.dropout(trans, 0.5, seed=1)

        previous_nodes = nodes

    sys.stdout.write("\n")

    if one_hot:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(act, y_gold))
        y = tf.nn.softmax(act, name="y")
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(act, y_gold))
        y = tf.nn.sigmoid(act, name="y")

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def rnn_model(
        features,
        notes,
        steps,
        hidden,
        graph_type,
        learning_rate):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, steps, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, steps, notes], name="y_gold")
    initial_state = rnn_initial_state(graph_type, hidden)

    weights = {
        'hidden': tf.Variable(tf.random_normal([features, hidden])),
        'out': tf.Variable(tf.random_normal([hidden, notes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden])),
        'out': tf.Variable(tf.random_normal([notes]))
    }

    def make_graph(_x, _weights, _biases):                            # (batch, steps, feats)
        _x = tf.transpose(_x, [1, 0, 2])                              # (steps, batch, feats)
        _x = tf.reshape(_x, [-1, features])                           # (steps * batch, feats)
        _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden']    # (steps * batch, hidden)
        _x = tf.split(0, steps, _x)                                   # (steps, batch, hidden)

        cell = rnn_cell(graph_type, hidden)                           # (steps, batch, hidden)

        output = rnn(graph_type, cell, initial_state, _x)             # (steps, batch, hidden)
        output = tf.reshape(output, [-1, hidden])                     # (steps * batch, hidden)
        output = tf.matmul(output, _weights["out"]) + _biases["out"]  # (steps * batch, notes)
        output = tf.split(0, steps, output)                           # (steps, batch, notes)

        return tf.transpose(output, [1, 0, 2])                        # (batch, steps, notes)

    log_y = make_graph(x, weights, biases)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(log_y, y_gold))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    y = tf.sigmoid(log_y)

    return Model(x, y, y_gold, loss, train_step, initial_state)


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

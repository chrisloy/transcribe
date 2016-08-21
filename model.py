import sys
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, x, y, y_gold, loss, train_step, initial_state=None):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step
        self.report_name = "ERROR"
        self.report_target = loss
        self.initial_state = initial_state

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


def time_series(learning_rate):

    tf.set_random_seed(1)

    sequence_length = 50
    features = 660
    hidden_size = 88
    output_notes = 88

    x = tf.placeholder(tf.float32, shape=[None, sequence_length, features], name="x")

    # xi = tf.transpose(x, [1, 0, 2])
    # xi = tf.reshape(xi, [-1, features])

    y_gold = tf.placeholder(tf.float32, shape=[None, sequence_length, output_notes], name="y_gold")

    w = param_norm([hidden_size, features], "W")
    b = param_norm([hidden_size, 1], "b")

    # (?, 1, 660)
    xs = tf.split(1, sequence_length, x)
    xs = tf.squeeze(xs)
    hs = []

    for i in range(sequence_length):
        # xi = tf.transpose(xi, [1, 0, 2])
        # xi = tf.reshape(xi, [-1, features])
        xi = xs[i]
        hs += tf.matmul(w, xi) + b

    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    initial_state = cell.zero_state(sequence_length, tf.float32)
    output, _ = tf.nn.rnn(cell, hs, initial_state=initial_state)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y_gold))

    y = tf.nn.sigmoid(output, name="y")
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def rnn(learning_rate):

    tf.set_random_seed(1)

    n_steps = 50
    n_input = 660
    n_hidden = 88
    n_classes = 88

    x = tf.placeholder(tf.float32, shape=[None, n_steps, n_input], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, n_steps, n_classes], name="y_gold")
    initial_state = tf.placeholder(tf.float32, shape=[None, 2 * n_hidden], name="initial_state")

    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def make_graph(_x, _weights, _biases):
        _x = tf.transpose(_x, [1, 0, 2])
        _x = tf.reshape(_x, [-1, n_input])  # (n_steps*batch_size, n_input)
        _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden']

        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        _x = tf.split(0, n_steps, _x)

        outputs, states = tf.nn.rnn(cell, _x, initial_state=initial_state)

        # result = []
        #
        # for i in range(n_steps):
        #     result += tf.matmul(outputs[i], _weights['out']) + _biases['out']

        return outputs

    y = make_graph(x, weights, biases)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_gold))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return Model(x, y, y_gold, loss, train_step, initial_state)


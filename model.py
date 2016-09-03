import graphs
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


def feed_forward_model(
        features,
        output,
        learning_rate=0.001,
        hidden_nodes=list(),
        dropout=None,
        one_hot=False):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    act = graphs.deep_neural_network(x, [features] + hidden_nodes + [output], dropout)
    y, loss = y_and_loss(act, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate))


def rnn_model(
        features,
        notes,
        steps,
        hidden,
        graph_type,
        learning_rate,
        one_hot=False):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, steps, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, steps, notes], name="y_gold")
    logits, i_state = graphs.recurrent_neural_network(x, features, notes, steps, hidden, graph_type)
    y, loss = y_and_loss(logits, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate), i_state)


def hybrid_model(
        features,
        notes,
        steps,
        rnn_input,
        rnn_width,
        hidden_nodes,
        rnn_type,
        learning_rate,
        dropout=None,
        one_hot=False):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, notes], name="y_gold")

    acoustic = graphs.deep_neural_network(x, [features] + hidden_nodes + [rnn_input], dropout)
    sequence, i_state = graphs.recurrent_neural_network(acoustic, rnn_input, notes, steps, rnn_width, rnn_type)

    y, loss = y_and_loss(sequence, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate), i_state)


def train(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def y_and_loss(logits, y_gold, one_hot):
    if one_hot:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.softmax(logits, name="y")
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.sigmoid(logits, name="y")
    return y, loss

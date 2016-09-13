import graphs
import tensorflow as tf
from functools import partial


class Model:
    def __init__(self, x, y, y_gold, loss, train_step, i_state=None, pre_loss=None, pre_train=None):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step
        self.report_name = "ERROR"
        self.report_target = loss
        self.i_state = i_state
        self.pre_loss = pre_loss
        self.pre_train = pre_train

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
    act, _ = graphs.deep_neural_network(x, [features] + hidden_nodes + [output], dropout)
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
    logits = graphs.recurrent_neural_network(x, features, notes, steps, hidden, graph_type)
    y, loss = y_and_loss(logits, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate))


def hybrid_model(
        features,
        notes,
        steps,
        rnn_state_size,
        acoustic_hidden_nodes,
        rnn_type,
        rnn_learning_rate,
        acoustic_learning_rate,
        dropout=None,
        one_hot=False,
        freeze_frame_model=True):

    tf.set_random_seed(1)

    assert acoustic_hidden_nodes is not None

    x = tf.placeholder(tf.float32, shape=[None, steps, features], name="x_sequence")
    y_gold = tf.placeholder(tf.float32, shape=[None, steps, notes], name="y_sequence_gold")

    x_acoustic = x                                                        # (batch, steps, features)
    x_acoustic = tf.transpose(x_acoustic, [1, 0, 2])                      # (steps, batch, features)
    x_acoustic = tf.reshape(x_acoustic, [-1, features])                   # (steps * batch, features)

    y_acoustic_gold = y_gold                                              # (batch, steps, notes)
    y_acoustic_gold = tf.transpose(y_acoustic_gold, [1, 0, 2])            # (steps, batch, notes)
    y_acoustic_gold = tf.reshape(y_acoustic_gold, [-1, notes])            # (steps * batch, notes)

    # Acoustic Model
    layers = [features] + acoustic_hidden_nodes

    logits_acoustic_fixed, _ = graphs.deep_neural_network(x_acoustic, layers, dropout)
    logits_acoustic = graphs.logistic_regression(logits_acoustic_fixed, acoustic_hidden_nodes[-1], notes)
    y_acoustic, loss_acoustic = y_and_loss(logits_acoustic, y_acoustic_gold, one_hot)

    if freeze_frame_model:
        frozen_acoustic = tf.stop_gradient(logits_acoustic_fixed)
    else:
        frozen_acoustic = logits_acoustic_fixed

    train_acoustic = train(loss_acoustic, acoustic_learning_rate)
    acoustic = Model(x, y_acoustic, y_gold, loss_acoustic, train_acoustic)

    def transfer_layer(not_used):
        return graphs.logistic_regression(frozen_acoustic, acoustic_hidden_nodes[-1], rnn_state_size)

    # Sequence Model
    sequence = graphs.recurrent_neural_network(
        x,
        acoustic_hidden_nodes[-1],
        notes,
        steps,
        rnn_state_size,
        rnn_type,
        input_model=transfer_layer
    )

    y_sequence, loss_sequence = y_and_loss(sequence, y_gold, one_hot)
    train_sequence = train(loss_sequence, rnn_learning_rate)

    sequence = Model(x, y_sequence, y_gold, loss_sequence, train_sequence)

    return acoustic, sequence


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

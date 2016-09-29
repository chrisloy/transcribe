import graphs
import tensorflow as tf


class Model:
    def __init__(self, x, y, y_gold, loss, train_step, i_state=None, pre_loss=None, pre_train=None, training=None):
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
        self.training = training

    def set_report(self, name, target):
        self.report_name = name
        self.report_target = target

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_gold, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def dev_labelled_feed(self, d):
        l = d.x_test.shape[0] / 2
        if self.training is not None:
            return {self.x: d.x_train[0:l, :], self.y_gold: d.y_train[0:l, :], self.training: False}
        else:
            return {self.x: d.x_train[0:l, :], self.y_gold: d.y_train[0:l, :]}

    def test_labelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_test, self.y_gold: d.y_test, self.training: False}
        else:
            return {self.x: d.x_test, self.y_gold: d.y_test}

    def train_unlabelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_train, self.training: False}
        else:
            return {self.x: d.x_train}

    def test_unlabelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_test, self.training: False}
        else:
            return {self.x: d.x_test}

    def train_batch_feed(self, d, lower, upper):
        if self.training is not None:
            return {self.x: d.x_train[lower:upper], self.y_gold: d.y_train[lower:upper], self.training: True}
        else:
            return {self.x: d.x_train[lower:upper], self.y_gold: d.y_train[lower:upper]}


def feed_forward_model(
        features,
        output,
        learning_rate=0.001,
        hidden_nodes=list(),
        dropout=None,
        one_hot=False,
        batch_norm=False):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    training = tf.placeholder(tf.bool) if batch_norm else None
    act, _ = graphs.deep_neural_network(
        x,
        [features] + hidden_nodes + [output],
        dropout=dropout,
        batch_norm=batch_norm,
        training=training
    )
    y, loss = y_and_loss(act, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate), training=training)


def train(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def y_and_loss(logits, y_gold, one_hot=False):
    if one_hot:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.softmax(logits, name="y")
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.sigmoid(logits, name="y")
    return y, loss


def frame_model(
        graph_type,
        features,
        output,
        learning_rate=0.02,
        hidden_nodes=list(),
        dropout=None,
        one_hot=False):
    if graph_type == 'mlp':
        return feed_forward_model(features, output, learning_rate, hidden_nodes, dropout, one_hot)
    elif graph_type == 'ladder':
        return ladder_model(features, output, learning_rate, hidden_nodes)


def ladder_model(
        features,
        output,
        learning_rate,
        hidden_nodes):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    training = tf.placeholder(tf.bool, name="training")

    layers = [features] + hidden_nodes + [output]
    noise_var = 0.1
    noise_costs = [0.1, 0.01, 0.01, 0.01]
    y_clean, y_corr, u_cost = graphs.ladder_network(x, layers, noise_var, training, denoising_cost=noise_costs)
    y, s_cost = y_and_loss(y_clean, y_gold)
    loss = s_cost + u_cost
    train_step = train(loss, learning_rate)

    m = Model(x, y, y_gold, loss, train_step, training=training)
    m.set_report("ERROR", s_cost)
    return m


def hierarchical_deep_network(
        features,
        notes,
        steps,
        frame_hidden_nodes,
        frame_dropout,
        frame_learning_rate,
        sequence_hidden_nodes,
        sequence_dropout,
        sequence_learning_rate):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, steps, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, steps, notes], name="y_gold")

    x_frame = tf.reshape(x, [-1, features])
    y_frame_gold = tf.reshape(y_gold, [-1, notes])

    # Frame model
    frame_layers = [features] + frame_hidden_nodes + [notes]
    logits_frame, _ = graphs.deep_neural_network(x_frame, frame_layers, frame_dropout)
    y_frame, loss_frame = y_and_loss(logits_frame, y_frame_gold)
    frozen_frame = tf.stop_gradient(logits_frame)
    train_frame = train(loss_frame, frame_learning_rate)
    frame = Model(x, y_frame, y_gold, loss_frame, train_frame)

    x_sequence = tf.reshape(frozen_frame, [-1, steps * notes])

    # Sequence model
    sequence_layers = [steps * notes] + sequence_hidden_nodes + [steps * notes]
    logits_sequence, _ = graphs.deep_neural_network(x_sequence, sequence_layers, sequence_dropout, init='identity')
    y_gold_flat = tf.reshape(y_gold, [-1, steps * notes])
    y_sequence, loss_sequence = y_and_loss(logits_sequence, y_gold_flat)
    y_sequence = tf.reshape(y_sequence, [-1, steps, notes])
    train_sequence = train(loss_sequence, sequence_learning_rate)
    sequence = Model(x, y_sequence, y_gold, loss_sequence, train_sequence)

    return frame, sequence


def hierarchical_recurrent_network(
        features,
        notes,
        steps,
        frame_hidden_nodes,
        frame_dropout,
        frame_learning_rate,
        rnn_graph_type,
        rnn_learning_rate):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, steps, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, steps, notes], name="y_gold")

    x_frame = tf.reshape(x, [-1, features])
    y_frame_gold = tf.reshape(y_gold, [-1, notes])

    # Frame model
    frame_layers = [features] + frame_hidden_nodes + [notes]
    logits_frame, hs = graphs.deep_neural_network(x_frame, frame_layers, frame_dropout)
    y_frame, loss_frame = y_and_loss(logits_frame, y_frame_gold)
    train_frame = train(loss_frame, frame_learning_rate)
    frame = Model(x, y_frame, y_gold, loss_frame, train_frame)

    rnn_size = frame_hidden_nodes[-1]
    frozen_frame = tf.stop_gradient(hs[-1])
    x_rnn = tf.reshape(frozen_frame, [-1, steps, rnn_size])

    # Sequence model
    logits_rnn = graphs.recurrent_neural_network(x_rnn, rnn_size, notes, steps, rnn_graph_type)
    y_gold_flat = tf.reshape(y_gold, [-1, steps * notes])
    y_rnn, loss_rnn = y_and_loss(tf.reshape(logits_rnn, [-1, steps * notes]), y_gold_flat)
    y_rnn = tf.reshape(y_rnn, [-1, steps, notes])
    train_rnn = train(loss_rnn, rnn_learning_rate)
    rnn = Model(x, y_rnn, y_gold, loss_rnn, train_rnn)

    return frame, rnn

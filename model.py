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
    act = graphs.deep_neural_network(
        x,
        [features] + hidden_nodes + [output],
        dropout=dropout,
        batch_norm=batch_norm,
        training=training
    )
    y, loss = y_and_loss(act, y_gold, one_hot)

    return Model(x, y, y_gold, loss, train(loss, learning_rate), training=training)


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
    x_acoustic = split_sequence_into_frames(x, features)
    y_acoustic_gold = split_sequence_into_frames(y_gold, notes)

    # Acoustic Model
    layers = [features] + acoustic_hidden_nodes

    logits_acoustic_fixed = graphs.deep_neural_network(x_acoustic, layers, dropout)
    logits_acoustic = graphs.logistic_regression(logits_acoustic_fixed, acoustic_hidden_nodes[-1], notes)
    y_acoustic, loss_acoustic = y_and_loss(logits_acoustic, y_acoustic_gold, one_hot)

    if freeze_frame_model:
        frozen_acoustic = tf.stop_gradient(logits_acoustic_fixed)
    else:
        frozen_acoustic = logits_acoustic_fixed

    train_acoustic = train(loss_acoustic, acoustic_learning_rate)
    acoustic = Model(x, y_acoustic, y_gold, loss_acoustic, train_acoustic)

    def transfer_layer(_):
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


def hybrid_model_no_transfers(
        features,
        notes,
        steps,
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
    x_acoustic = split_sequence_into_frames(x, features)
    y_acoustic_gold = split_sequence_into_frames(y_gold, notes)

    # Acoustic Model
    layers = [features] + acoustic_hidden_nodes + [notes]

    logits_acoustic_fixed = graphs.deep_neural_network(x_acoustic, layers, dropout)
    logits_acoustic = logits_acoustic_fixed
    y_acoustic, loss_acoustic = y_and_loss(logits_acoustic, y_acoustic_gold, one_hot)

    if freeze_frame_model:
        frozen_acoustic = tf.stop_gradient(logits_acoustic_fixed)
    else:
        frozen_acoustic = logits_acoustic_fixed

    train_acoustic = train(loss_acoustic, acoustic_learning_rate)
    acoustic = Model(x, y_acoustic, y_gold, loss_acoustic, train_acoustic)

    def transfer_layer(_):
        return frozen_acoustic

    # Sequence Model
    sequence = graphs.rnn_no_layers(
        x,
        notes,
        steps,
        rnn_type,
        transfer_layer
    )

    y_sequence, loss_sequence = y_and_loss(sequence, y_gold, one_hot)
    train_sequence = train(loss_sequence, rnn_learning_rate)

    sequence = Model(x, y_sequence, y_gold, loss_sequence, train_sequence)

    return acoustic, sequence


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
    logits_frame = graphs.deep_neural_network(x_frame, frame_layers, frame_dropout)
    y_frame, loss_frame = y_and_loss(logits_frame, y_frame_gold)
    frozen_frame = tf.stop_gradient(logits_frame)
    train_frame = train(loss_frame, frame_learning_rate)
    frame = Model(x, y_frame, y_gold, loss_frame, train_frame)

    x_sequence = tf.reshape(frozen_frame, [-1, steps * notes])

    # Sequence model
    sequence_layers = [steps * notes] + sequence_hidden_nodes + [steps * notes]
    logits_sequence = graphs.deep_neural_network(x_sequence, sequence_layers, sequence_dropout, init='identity')
    y_gold_flat = tf.reshape(y_gold, [-1, steps * notes])
    y_sequence, loss_sequence = y_and_loss(logits_sequence, y_gold_flat)
    y_sequence = tf.reshape(y_sequence, [-1, steps, notes])
    train_sequence = train(loss_sequence, sequence_learning_rate)
    sequence = Model(x, y_sequence, y_gold, loss_sequence, train_sequence)

    return frame, sequence


def split_sequence_into_frames(sequence, frame_size):  # (batch, steps, frame_size)
    frame = tf.transpose(sequence, [1, 0, 2])          # (steps, batch, frame_size)
    return tf.reshape(frame, [-1, frame_size])         # (steps * batch, frame_size)

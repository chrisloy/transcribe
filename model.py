import graphs
import tensorflow as tf
from tensorflow.python import control_flow_ops
import math


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

    def train_labelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_train, self.y_gold: d.y_train, self.training: False}
        else:
            return {self.x: d.x_train, self.y_gold: d.y_train}

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
            return {self.x: d.x_train[lower:upper, :], self.y_gold: d.y_train[lower:upper, :], self.training: True}
        else:
            return {self.x: d.x_train[lower:upper, :], self.y_gold: d.y_train[lower:upper, :]}


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

    x_acoustic = x                                              # (batch, steps, features)
    x_acoustic = tf.transpose(x_acoustic, [1, 0, 2])            # (steps, batch, features)
    x_acoustic = tf.reshape(x_acoustic, [-1, features])         # (steps * batch, features)

    y_acoustic_gold = y_gold                                    # (batch, steps, notes)
    y_acoustic_gold = tf.transpose(y_acoustic_gold, [1, 0, 2])  # (steps, batch, notes)
    y_acoustic_gold = tf.reshape(y_acoustic_gold, [-1, notes])  # (steps * batch, notes)

    # Acoustic Model
    layers = [features] + acoustic_hidden_nodes + [notes]

    logits_acoustic_fixed, _ = graphs.deep_neural_network(x_acoustic, layers, dropout)
    logits_acoustic = logits_acoustic_fixed
    y_acoustic, loss_acoustic = y_and_loss(logits_acoustic, y_acoustic_gold, one_hot)

    if freeze_frame_model:
        frozen_acoustic = tf.stop_gradient(logits_acoustic_fixed)
    else:
        frozen_acoustic = logits_acoustic_fixed

    train_acoustic = train(loss_acoustic, acoustic_learning_rate)
    acoustic = Model(x, y_acoustic, y_gold, loss_acoustic, train_acoustic)

    def transfer_layer(not_used):
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


def y_and_loss(logits, y_gold, one_hot):
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
        batch_size,
        learning_rate=0.02,
        hidden_nodes=list(),
        dropout=None,
        one_hot=False):
    if graph_type == 'mlp':
        return feed_forward_model(features, output, learning_rate, hidden_nodes, dropout, one_hot)
    elif graph_type == 'ladder':
        return ladder_model(features, output, batch_size, learning_rate, hidden_nodes)


def ladder_model(
        features,
        output,
        batch_size,
        learning_rate=0.02,
        hidden_nodes=list()):

    # hyperparameters that denote the importance of each layer
    denoising_cost = [100.0, 1.0, 0.10, 0.10, 0.10, 0.10, 0.10]

    # Adapted from https://github.com/rinuboney/ladder

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")

    layer_sizes = [features] + hidden_nodes + [output]

    ls = len(layer_sizes) - 1  # number of layers

    def bi(inits, size, name):
        return tf.Variable(inits * tf.ones([size]), name=name)

    def wi(shape, name):
        return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

    shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers

    weights = {
        'W':     [wi(s, "W") for s in shapes],                             # Encoder weights
        'V':     [wi(s[::-1], "V") for s in shapes],                       # Decoder weights
        'beta':  [bi(0.0, layer_sizes[l + 1], "beta") for l in range(ls)],  # batch norm param to shift the norm'd value
        'gamma': [bi(1.0, layer_sizes[l + 1], "beta") for l in range(ls)]   # batch norm param to scale the norm'd value
    }

    noise_std = 0.1  # scaling factor for noise used in corrupted encoder

    def join(a, b):
        return tf.concat(0, [a, b])

    def labeled(q):
        return tf.slice(q, [0, 0], [batch_size, -1]) if q is not None else q

    def unlabeled(q):
        return tf.slice(q, [batch_size, 0], [-1, -1]) if q is not None else q

    def split_lu(q):
        return labeled(q), unlabeled(q)

    training = tf.placeholder(tf.bool)

    ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
    bn_assigns = []  # this list stores the updates to be made to average mean and variance

    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    # average mean and variance of all layers
    running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
    running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

    def update_batch_normalization(batch, l):
        """batch normalize + update average mean and variance of layer l"""
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = running_mean[l - 1].assign(mean)
        assign_var = running_var[l - 1].assign(var)
        bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def encoder(inputs, noise_std):
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
        # to store the pre-activation, activation, mean and variance for each layer
        d = {
            'labeled': {'z': {}, 'm': {}, 'v': {}, 'h': {}},
            'unlabeled': {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        }
        # The data for labeled and unlabeled examples are stored separately
        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
        for l in range(1, ls + 1):
            print "Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l]
            d['labeled']['h'][l - 1], d['unlabeled']['h'][l - 1] = split_lu(h)
            z_pre = tf.matmul(h, weights['W'][l - 1])  # pre-activation
            z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

            m, v = tf.nn.moments(z_pre_u, axes=[0])

            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                    z += tf.random_normal(tf.shape(z_pre)) * noise_std
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using
                    # batch mean and variance of labeled examples
                    z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
                return z

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = ewma.average(running_mean[l - 1])
                var = ewma.average(running_var[l - 1])
                z = batch_normalization(z_pre, mean, var)
                # Instead of the above statement, the use of the following 2 statements containing a typo
                # consistently produces a 0.2% higher accuracy for unclear reasons.
                # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = control_flow_ops.cond(training, training_batch_norm, eval_batch_norm)

            if l == ls:
                # use softmax activation in output layer
                h = tf.nn.softmax(weights['gamma'][l - 1] * (z + weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + weights["beta"][l - 1])
            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # mean and var of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
        return h, d

    print "=== Corrupted Encoder ==="
    y_c, corr = encoder(x, noise_std)

    print "=== Clean Encoder ==="
    y, clean = encoder(x, 0.0)  # 0.0 -> do not add noise

    print "=== Decoder ==="

    # Decoder
    z_est = {}
    d_cost = []  # to store the denoising cost of all layers
    for l in range(ls, -1, -1):
        print "Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[
            l], ", denoising cost: ", denoising_cost[l]
        z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
        m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1 - 1e-10)
        if l == ls:
            u = unlabeled(y_c)
        else:
            u = tf.matmul(z_est[l + 1], weights['V'][l])
        u = batch_normalization(u)
        z_est[l] = g_gauss(z_c, u, layer_sizes[l])
        z_est_bn = (z_est[l] - m) / v
        # append the cost of this layer to d_cost
        d_cost.append(
            (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    # calculate total unsupervised cost by adding the denoising cost of all layers
    u_cost = tf.add_n(d_cost)

    # y_n = labeled(y_c)

    # cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_N), 1))  # supervised cost

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_c, y_gold))

    loss = cost + u_cost  # total cost

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    return Model(x, y, y_gold, loss, train_step, training=training)


def g_gauss(z_c, u, size):
    """gaussian denoising function proposed in the original paper"""

    def wi(inits, name):
        return tf.Variable(inits * tf.ones([size]), name=name)

    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est


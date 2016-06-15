import tensorflow as tf


class Model:
    def __init__(self, x, y, y_gold, loss, train_step):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step


def param_norm(shape):
    return tf.Variable(tf.random_normal(shape, 0.35), dtype="float32")


def param_zeros(shape):
    return tf.Variable(tf.zeros(shape), dtype="float32")


def feed_forward_model(features, hidden_nodes, output):

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_gold = tf.placeholder(tf.float32, shape=[None, output])
    previous_nodes = features
    h = x

    for nodes in hidden_nodes + [output]:
        w = param_zeros([previous_nodes, nodes])
        b = param_zeros([nodes])
        h = tf.nn.sigmoid(tf.matmul(h, w) + b)
        previous_nodes = nodes

    y = h

    loss = tf.reduce_mean(tf.square(y - y_gold))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def logistic_regression(features):

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_gold = tf.placeholder(tf.float32, shape=[None, 2])

    w = param_zeros([features, 2])
    b = param_zeros([2])
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    loss = tf.reduce_mean(-tf.reduce_sum(y_gold * tf.log(tf.clip_by_value(y, 1e-20, 1.0)), reduction_indices=1))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


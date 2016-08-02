import sys
import tensorflow as tf


class Model:
    def __init__(self, x, y, y_gold, loss, train_step):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step
        self.report_name = "ERROR"
        self.report_target = loss

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


def feed_forward_model(features, output, learning_rate=0.001, hidden_nodes=list(), loss_function="mse", dropout=False):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    previous_nodes = features
    act = None
    trans = x

    depth = 0

    sys.stdout.write("Graph shape: %d" % previous_nodes)

    for nodes in hidden_nodes + [output]:
        sys.stdout.write(" --> %d" % nodes)
        depth += 1
        w = param_norm([previous_nodes, nodes], "W%d" % depth)
        b = param_norm([nodes], "b%d" % depth)
        act = tf.matmul(trans, w) + b
        trans = tf.nn.sigmoid(act)
        # trans = tf.nn.relu(act)
        if dropout:
            trans = tf.nn.dropout(trans, 0.5, seed=1)  # 0.8? 0.5?
        previous_nodes = nodes

    sys.stdout.write("\n")

    y = tf.nn.softmax(act, name="y")  # TODO
    loss = get_loss_function(loss_function, y, y_gold)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def get_loss_function(loss_function, y, y_gold):
    if loss_function == "mse":
        return tf.reduce_mean(tf.square(y - y_gold))
    elif loss_function == "absolute":
        return tf.reduce_mean(tf.abs(y - y_gold))
    elif loss_function == "cross_entropy":
        return tf.reduce_mean(-tf.reduce_sum(y_gold * tf.log(tf.clip_by_value(y, 1e-20, 1.0)), reduction_indices=1))
    else:
        raise NameError("Unknown loss function %s" % loss_function)

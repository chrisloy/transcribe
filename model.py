import tensorflow as tf


class Model:
    def __init__(self, x, y, y_gold, loss, train_step):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step


def param_norm(shape):
    return tf.Variable(tf.truncated_normal(shape, 0.1), dtype="float32")


def param_zeros(shape):
    return tf.Variable(tf.zeros(shape), dtype="float32")


def feed_forward_model(features, output, learning_rate=0.001, hidden_nodes=list(), loss_function="mse", dropout=False):

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_gold = tf.placeholder(tf.float32, shape=[None, output])
    previous_nodes = features
    h = x

    for nodes in hidden_nodes + [output]:
        w = param_zeros([previous_nodes, nodes])
        b = param_zeros([nodes])
        h = tf.nn.sigmoid(tf.matmul(h, w) + b)
        if dropout:
            h = tf.nn.dropout(h, 0.7)  # 0.8? 0.5?
        previous_nodes = nodes

    y = h
    loss = get_loss_function(loss_function, y, y_gold)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def get_loss_function(loss_function, y, y_gold):
    if loss_function == "mse":
        return tf.reduce_mean(tf.square(y - y_gold))
    elif loss_function == "cross_entropy":
        return tf.reduce_mean(-tf.reduce_sum(y_gold * tf.log(tf.clip_by_value(y, 1e-20, 1.0)), reduction_indices=1))
    else:
        raise NameError("Unknown loss function %s" % loss_function)

import generate
import midi
import numpy as np
import preprocess
import slicer
import spectrogram
import sys
import tensorflow as tf
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from multiprocessing import Pool
from os import devnull


class Data:

    def __init__(self, x_train, y_train, x_test, y_test, batches, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batches = batches
        self.batch_size = batch_size
        self.features = x_train.shape[1]

    def to_one_hot(self):
        # Change y labels into one-hot vectors in two dimensions.
        y_train = np.stack([1 - self.y_train, self.y_train], axis=2)
        y_test = np.stack([1 - self.y_test, self.y_test], axis=2)
        return Data(self.x_train, y_train, self.x_test, y_test, self.batches, self.batch_size)

    def to_note(self, n):
        return Data(self.x_train, self.y_train[:, n, :], self.x_test, self.y_test[:, n, :], self.batches,
                    self.batch_size)


class Model:
    def __init__(self, x, y, y_gold, loss, train_step):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step


def data_to_x_and_slices(data):
    return data.astype("float32"), data.shape[1]


def load_x(wav_file, engine):
    data = spectrogram.spectrogram_cqt(wav_file, engine)
    return data_to_x_and_slices(data)


def load_cached_x(cache_file):
    data = preprocess.refresh(cache_file)
    return data_to_x_and_slices(data)


def load_y(midi_file, slices):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)


def load_pair(i, engine):
    xi, s = load_x("corpus/%04d.wav" % i, engine)
    yi = load_y("corpus/%04d.mid" % i, s)
    return xi, yi


def load_pair_from_cache(i):
    xi, s = load_cached_x("corpus/%04d_features.p" % i)
    yi = load_y("corpus/%04d.mid" % i, s)
    return xi, yi


def load_slices(a, b, slice_samples, from_cache):

    pool = Pool(processes=8)

    x = []
    y = []

    if from_cache:
        xys = pool.map(load_pair_from_cache, range(a, b))
        for xi, yi in xys:
            x.append(xi)
            y.append(yi)

    else:
        engine = spectrogram.cqt_engine(slice_samples, 60)
        for i in range(a, b):
            sys.stdout.write("%d/%d\r" % (i - a, b - a))
            sys.stdout.flush()
            xi, yi = load_pair(i, engine)
            x.append(xi)
            y.append(yi)

    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)

    return np.transpose(x), np.transpose(y)


def load_data(train_size, test_size, slice_samples, from_cache, batch_size):
    print "Loading training set...."
    x_train, y_train = load_slices(0, train_size, slice_samples, from_cache)
    print "Loading testing set...."
    x_test, y_test = load_slices(1000 - test_size, 1000, slice_samples, from_cache)
    batches = x_train.shape[0] / batch_size
    return Data(x_train, y_train, x_test, y_test, batches, batch_size)


def param_norm(shape):
    return tf.Variable(tf.random_normal(shape, 0.35), dtype="float32")


def param_zeros(shape):
    return tf.Variable(tf.zeros(shape), dtype="float32")


def train_model(epochs, model, data):
    for j in range(epochs):
        for k in range(data.batches):
            sys.stdout.write("EPOCH %02d/%d - BATCH %03d/%d\r" % (j + 1, epochs, k + 1, data.batches))
            sys.stdout.flush()

            start = k * data.batch_size
            stop = (k + 1) * data.batch_size

            model.train_step.run(feed_dict={
                model.x:      data.x_train[start:stop, :],
                model.y_gold: data.y_train[start:stop, :]
            })

        if j + 1 == epochs or (j + 1) % 20 == 0:
            sys.stdout.write("EPOCH %02d/%d - TRAIN ERROR: %0.16f - TEST ERROR: %0.16f\n" %
                             (
                                 j + 1,
                                 epochs,
                                 model.loss.eval(feed_dict={model.x: data.x_train, model.y_gold: data.y_train}),
                                 model.loss.eval(feed_dict={model.x: data.x_test, model.y_gold: data.y_test})
                             ))
            sys.stdout.flush()


def feed_forward_model(features):

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_gold = tf.placeholder(tf.float32, shape=[None, 128])

    w1 = param_zeros([features, features])
    b1 = param_zeros([features])
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = param_zeros([features, features])
    b2 = param_zeros([features])
    h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

    w3 = param_zeros([features, 128])
    b3 = param_zeros([128])
    h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

    y = h3

    loss = tf.reduce_mean(tf.square(y - y_gold))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    return Model(x, y, y_gold, loss, train_step)


def run_joint_model(epochs, train_size, test_size, slice_samples=512, batch_size=1000, from_cache=True):
    sess = tf.InteractiveSession()
    data = load_data(train_size, test_size, slice_samples, from_cache, batch_size)
    model = feed_forward_model(data.features)
    sess.run(tf.initialize_all_variables())

    train_model(epochs, model, data)

    y_pred = model.y.eval(feed_dict={model.x: data.x_test})
    fpr, tpr, thresholds = roc_curve(data.y_test.flatten(), y_pred.flatten())

    plt.plot(fpr, tpr)
    plt.show()


def produce_prediction(slice_samples, x, y):
    ex, _ = load_x("output/sanity.wav", slice_samples)
    ex = np.transpose(ex)

    output = np.transpose(y.eval(feed_dict={x: ex}))
    print output.shape
    track = slicer.boolean_table_to_track(output)

    print track
    original_track = midi.read_midifile("output/sanity.mid")[0]
    print original_track

    pattern = midi.Pattern()
    pattern.append(track)
    midi.write_midifile("output/sanity_pred_deep.mid", pattern)
    generate.write_wav_file("output/sanity_pred.mid", "output/sanity_pred_deep.wav", open(devnull, 'w'))


def run_individual_classifiers(epochs, train_size, test_size, slice_samples=512, batch_size=1000, from_cache=True,
                               notes=range(128)):
    sess = tf.InteractiveSession()

    start_note = min(notes)
    max_notes = len(notes)

    data = load_data(train_size, test_size, slice_samples, from_cache, batch_size).to_one_hot()

    models = []

    for i in range(len(notes)):
        x = tf.placeholder(tf.float32, shape=[None, data.features])
        y_gold = tf.placeholder(tf.float32, shape=[None, 2])
        w = param_zeros([data.features, 2])
        b = param_zeros([2])
        y = tf.nn.softmax(tf.matmul(x, w) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y_gold * tf.log(tf.clip_by_value(y, 1e-20, 1.0)), reduction_indices=1))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        models.append(Model(x, y, y_gold, loss, train_step))

    sess.run(tf.initialize_all_variables())

    for i in range(len(notes)):
        n = notes[i]
        print "NOTE %03d" % n
        train_model(epochs, models[i], data.to_note(n))

    y_pred = np.empty([data.y_test.shape[0], max_notes])

    for i in range(len(notes)):
        y_pred[:, i] = models[i].y.eval(feed_dict={models[i].x: data.x_test})[:, 1]
    fpr, tpr, thresholds = roc_curve(data.y_test[:, start_note:start_note+max_notes, 1].flatten(), y_pred.flatten())

    plt.plot(fpr, tpr)
    plt.show()


if __name__ == "__main__":
    # run_individual_classifiers(epochs=10, train_size=200, test_size=50, notes=range(67, 75))
    run_joint_model(epochs=10, train_size=20, test_size=10)

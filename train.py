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


def load_x(wav_file, engine):
    data = spectrogram.spectrogram_cqt(wav_file, engine)
    return data.astype("float32"), data.shape[1]


def load_cached_x(cache_file):
    data = preprocess.refresh(cache_file)
    return data.astype("float32"), data.shape[1]


def load_y(midi_file, slices):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)


def load_pair(i):

    # sys.stdout.write("%d/%d\r" % (i - a, b - a))
    # sys.stdout.flush()

    # xi, s = load_x("corpus/%04d.wav" % i, engine)
    xi, s = load_cached_x("corpus/%04d_features.p" % i)
    yi = load_y("corpus/%04d.mid" % i, s)

    return xi, yi


def load_slices(a, b, slice_samples):

    # engine = spectrogram.cqt_engine(slice_samples, 60)

    pool = Pool(processes=8)

    x = []
    y = []

    xys = pool.map(load_pair, range(a, b))

    for xi, yi in xys:
        x.append(xi)
        y.append(yi)

    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)

    return np.transpose(x), np.transpose(y)


def param_norm(shape):
    return tf.Variable(tf.random_normal(shape, 0.35), dtype="float32")


def param_zeros(shape):
    return tf.Variable(tf.zeros(shape), dtype="float32")


def run_logistic_regression():
    sess = tf.InteractiveSession()

    slice_samples = 4410
    features = 882

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_ = tf.placeholder(tf.float32, shape=[None, 128])

    w1 = param([features, 128])
    b1 = param([128])
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    # w2 = param([features, features])
    # b2 = param([features])
    # h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

    # w3 = param([features, 128])
    # b3 = param([128])
    # h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

    y = h1

    mse = tf.reduce_sum(tf.abs(y - y_))

    train_step = tf.train.AdamOptimizer().minimize(mse)

    sess.run(tf.initialize_all_variables())

    epochs = 2000

    print "Loading training set...."
    x_train, y_train = load_slices(0, 900, slice_samples)

    print "Loading testing set...."
    x_test, y_test = load_slices(900, 1000, slice_samples)

    batch_size = 1000
    batches = x_train.shape[0] / batch_size

    for j in range(epochs):

        for i in range(batches):

            sys.stdout.write("EPOCH %d/%d ... BATCH %d/%d\r" % (j + 1, epochs, i + 1, batches))
            sys.stdout.flush()

            start = i * batch_size
            stop = (i + 1) * batch_size

            train_step.run(feed_dict={x: x_train[start:stop, :], y_: y_train[start:stop, :]})

        if j % 20 == 0:
            sys.stdout.write("EPOCH %d  ...  TRAIN ERROR: %0.4f  ...  TEST ERROR: %0.4f\n" %
                             (
                                 j + 1,
                                 mse.eval(feed_dict={x: x_train, y_: y_train}),
                                 mse.eval(feed_dict={x: x_test, y_: y_test})
                             )
            )
            sys.stdout.flush()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()

    y_pred = y.eval(feed_dict={x: x_test})
    fpr, tpr, thresholds = roc_curve(y_test.flatten(), y_pred.flatten())

    plt.plot(fpr, tpr)
    plt.show()

    # ex, _ = load_x("output/sanity.wav", slice_samples)
    # ex = np.transpose(ex)
    #
    # output = np.transpose(y.eval(feed_dict={x: ex}))
    # print output.shape
    # track = slicer.boolean_table_to_track(output)
    #
    # print track
    # original_track = midi.read_midifile("output/sanity.mid")[0]
    # print original_track
    #
    # pattern = midi.Pattern()
    # pattern.append(track)
    # midi.write_midifile("output/sanity_pred_deep.mid", pattern)
    # generate.write_wav_file("output/sanity_pred.mid", "output/sanity_pred_deep.wav", open(devnull, 'w'))


def run_individual_classifiers():
    sess = tf.InteractiveSession()

    start_note = 67
    max_notes = 1
    slice_samples = 512
    features = 660
    epochs = 200
    notes = range(start_note, start_note + max_notes)

    print "Loading training set...."
    x_train, y_train = load_slices(0, 1000, slice_samples)

    print "Loading testing set...."
    x_test, y_test = load_slices(100, 1100, slice_samples)

    y_train_1h = np.stack([1 - y_train, y_train], axis=2)
    y_test_1h = np.stack([1 - y_test, y_test], axis=2)

    batch_size = 1000
    batches = x_train.shape[0] / batch_size

    x = tf.placeholder(tf.float32, shape=[None, features])

    y_ = []
    y = []
    loss = []
    train_step = []

    for n in notes:
        i = n - start_note
        y_.append(tf.placeholder(tf.float32, shape=[None, 2]))
        w = param_zeros([features, 2])
        b = param_zeros([2])
        y.append(tf.nn.softmax(tf.matmul(x, w) + b))
        # loss.append(tf.reduce_sum(tf.abs(y[n] - y_[n])))
        loss.append(tf.reduce_mean(-tf.reduce_sum(y_[i] * tf.log(tf.clip_by_value(y[i], 1e-20, 1.0)), reduction_indices=1)))
        train_step.append(tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss[i]))

    sess.run(tf.initialize_all_variables())

    for n in notes:

        i = n - start_note

        y_train_n = y_train_1h[:, i, :]
        y_test_n = y_test_1h[:, i, :]

        for j in range(epochs):

            for k in range(batches):

                sys.stdout.write("NOTE %03d - EPOCH %02d/%d - BATCH %03d/%d\r" % (n, j + 1, epochs, k + 1, batches))
                sys.stdout.flush()

                start = k * batch_size
                stop = (k + 1) * batch_size

                train_step[i].run(feed_dict={x: x_train[start:stop], y_[i]: y_train_n[start:stop]})

            if j + 1 == epochs or (j + 1) % 5 == 0:
                sys.stdout.write("NOTE %03d - EPOCH %02d/%d - TRAIN ERROR: %0.16f - TEST ERROR: %0.16f\n" %
                                 (
                                     n,
                                     j + 1,
                                     epochs,
                                     loss[i].eval(feed_dict={x: x_train, y_[i]: y_train_n}),
                                     loss[i].eval(feed_dict={x: x_test, y_[i]: y_test_n})
                                 ))
                sys.stdout.flush()

    y_pred = np.empty([y_test.shape[0], max_notes])

    for n in notes:
        i = n - start_note
        y_pred[:, i] = y[i].eval(feed_dict={x: x_test})[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[:, start_note:start_note+max_notes].flatten(), y_pred.flatten())

    plt.plot(fpr, tpr)
    plt.show()


if __name__ == "__main__":
    run_individual_classifiers()

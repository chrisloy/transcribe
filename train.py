import midi
import numpy as np
import slicer
import spectrogram
import sys
import tensorflow as tf
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def load_x(wav_file, slice_samples):
    _, ts, data = spectrogram.spectrogram_10hz(wav_file, slice_samples)
    return data, ts.shape[0]


def load_y(midi_file, slices):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)


def load_slices(a, b, slice_samples):
    x = []
    y = []
    for i in range(a, b):

        sys.stdout.write("%d/%d\r" % (i - a, b - a))
        sys.stdout.flush()

        xi, s = load_x("output/%04d.wav" % i, slice_samples)
        yi = load_y("output/%04d.mid" % i, s)

        if xi.shape[1] == yi.shape[1]:
            x.append(xi)
            y.append(yi)
        else:
            print "WARN: skipping", (xi.shape, yi.shape, i)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)
    return np.transpose(x), np.transpose(y)


def param(shape):
    return tf.Variable(tf.random_normal(shape, 0.35))


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

    # train_step = tf.train.GradientDescentOptimizer(2.5).minimize(mse)
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


if __name__ == "__main__":
    run_logistic_regression()

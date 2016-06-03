import generate
import midi
import numpy as np
from os import devnull
import slicer
import spectrogram
import sys
import tensorflow as tf


def load_x(wav_file, slices):
    _, _, data = spectrogram.load_slice(wav_file, slices)
    return data[:, 0:slices]  # TODO why do I have to trim the end?


def load_y(midi_file, slices):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)


def load_slices(a, b):
    x = []
    y = []
    for i in range(a, b):
        s = 2000
        yi = load_y("output/%04d.mid" % i, s)
        xi = load_x("output/%04d.wav" % i, s)
        if xi.shape[1] == yi.shape[1]:
            x.append(xi)
            y.append(yi)
        else:
            print "WARN: skipping", (xi.shape, yi.shape, i)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)
    return np.transpose(x), np.transpose(y)


def run_logistic_regression():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 163])
    y_ = tf.placeholder(tf.float32, shape=[None, 128])

    w1 = tf.Variable(tf.zeros([163, 163]))
    b1 = tf.Variable(tf.zeros([163]))
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.zeros([163, 163]))
    b2 = tf.Variable(tf.zeros([163]))
    h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

    w3 = tf.Variable(tf.zeros([163, 128]))
    b3 = tf.Variable(tf.zeros([128]))
    h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

    y = h3

    sess.run(tf.initialize_all_variables())

    mse = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(mse)

    for j in range(2):
        for i in range(400):

            sys.stdout.write("Batch %d %d\r" % (j, i))
            sys.stdout.flush()

            start = i*2
            stop = (i+1)*2
            x_train, y_train = load_slices(start, stop)

            train_step.run(feed_dict={x: x_train, y_: y_train})

            if i % 100 == 0:
                print "MSE: ", mse.eval(feed_dict={x: x_train, y_: y_train})

    x_test, y_test = load_slices(900, 1000)
    print "TEST MSE:", mse.eval(feed_dict={x: x_test, y_: y_test})

    ex = np.transpose(load_x("output/sanity.wav", 2000))

    output = np.transpose(y.eval(feed_dict={x: ex}))
    print output.shape
    track = slicer.boolean_table_to_track(output)

    print track
    original_track = midi.read_midifile("output/sanity.mid")[0]
    print original_track

    pattern = midi.Pattern()
    pattern.append(track)
    midi.write_midifile("output/sanity_pred.mid", pattern)
    generate.write_wav_file("output/sanity_pred.mid", "output/sanity_pred.wav", open(devnull, 'w'))


if __name__ == "__main__":
    run_logistic_regression()

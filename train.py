import data
import generate
import midi
import model
import numpy as np
import slicer
import sys
import tensorflow as tf
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from os import devnull


def train_model(epochs, m, d, report_epochs=10):
    for j in range(epochs + 1):
        if j == epochs or j % report_epochs == 0:
            sys.stdout.write("EPOCH %03d/%d - TRAIN %s: %0.16f - TEST %s: %0.16f\n" %
                             (
                                 j,
                                 epochs,
                                 m.report_name,
                                 m.report_target.eval(feed_dict={m.x: d.x_train, m.y_gold: d.y_train}),
                                 m.report_name,
                                 m.report_target.eval(feed_dict={m.x: d.x_test, m.y_gold: d.y_test})
                             ))
            sys.stdout.flush()

        if j < epochs:
            for k in range(d.batches):
                sys.stdout.write("EPOCH %03d/%d - BATCH %03d/%d\r" % (j + 1, epochs, k + 1, d.batches))
                sys.stdout.flush()

                start = k * d.batch_size
                stop = (k + 1) * d.batch_size

                m.train_step.run(feed_dict={
                    m.x:      d.x_train[start:stop, :],
                    m.y_gold: d.y_train[start:stop, :]
                })


def run_joint_model(epochs, train_size, test_size, slice_samples=512, batch_size=1000, from_cache=True,
                    corpus="corpus"):
    with tf.Session() as sess:
        d = data.load(train_size, test_size, slice_samples, from_cache, batch_size, corpus)
        m = model.feed_forward_model(d.features, 128, hidden_nodes=[d.features, d.features])
        sess.run(tf.initialize_all_variables())
        train_model(epochs, m, d)
        y_pred = m.y.eval(feed_dict={m.x: d.x_test}, session=sess)

    fpr, tpr, thresholds = roc_curve(d.y_test.flatten(), y_pred.flatten())
    plt.plot(fpr, tpr)
    plt.show()


def run_one_hot_joint_model(epochs, train_size, test_size, slice_samples=512, batch_size=1000, from_cache=True,
                            corpus="corpus"):
    with tf.Session() as sess:
        d = data.load(train_size, test_size, slice_samples, from_cache, batch_size, corpus).to_one_hot()
        m = model.feed_forward_model(
                d.features,
                89,
                hidden_nodes=[d.features * 2, d.features * 1.5, d.features, d.features * 0.5, d.features * 0.2],
                loss_function="cross_entropy",
                learning_rate=0.5)
        m.set_report("ACCURACY", m.accuracy())
        sess.run(tf.initialize_all_variables())
        train_model(epochs, m, d)

        y_pred = m.y.eval(feed_dict={m.x: d.x_test}, session=sess)

        fpr, tpr, thresholds = roc_curve(d.y_test.flatten(), y_pred.flatten())
        plt.plot(fpr, tpr)
        plt.show()


def produce_prediction(slice_samples, x, y):
    ex, _ = data.load_x("output/sanity.wav", slice_samples)
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
                               notes=range(128), corpus="corpus"):
    start_note = min(notes)
    max_notes = len(notes)

    d = data.load(train_size, test_size, slice_samples, from_cache, batch_size, corpus).to_binary_one_hot()

    with tf.Session() as sess:
        models = []

        for i in range(len(notes)):
            models.append(model.feed_forward_model(d.features, 2))

        sess.run(tf.initialize_all_variables())

        for i in range(len(notes)):
            n = notes[i]
            print "NOTE %03d" % n
            train_model(epochs, models[i], d.to_note(n))

        y_pred = np.empty([d.y_test.shape[0], max_notes])

        for i in range(len(notes)):
            y_pred[:, i] = models[i].y.eval(feed_dict={models[i].x: d.x_test}, session=sess)[:, 1]

    fpr, tpr, thresholds = roc_curve(d.y_test[:, start_note:start_note+max_notes, 1].flatten(), y_pred.flatten())

    plt.plot(fpr, tpr)
    plt.show()


if __name__ == "__main__":
    # run_individual_classifiers(epochs=200, train_size=300, test_size=100, notes=[67], corpus="mono_piano")
    run_one_hot_joint_model(epochs=100, train_size=400, test_size=100, corpus="mono_piano_simple")
    # run_joint_model(epochs=500, train_size=800, test_size=200)

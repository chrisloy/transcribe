import data
import generate
import midi
import model
import numpy as np
import persist
import slicer
import sys
import tensorflow as tf
import warnings
from collections import defaultdict
from domain import Params
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from os import devnull


def train_model(epochs, m, d, report_epochs=10):
    print "Training model with %d features..." % d.features
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


def run_joint_model(p, from_cache=True):
    with tf.Session() as sess:
        d = data.load(
            p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
        ).to_padded(p.padding).to_shuffled()
        m = model.feed_forward_model(
            d.features,
            p.outputs(),
            loss_function="absolute",
            hidden_nodes=p.hidden_nodes,
            learning_rate=p.learning_rate,
            dropout=True
        )

        sess.run(tf.initialize_all_variables())
        train_model(p.epochs, m, d, report_epochs=50)

        persist.save(sess, m, d, p)

        print "TRAIN"
        y_pred = m.y.eval(feed_dict={m.x: d.x_train}, session=sess)
        report_poly_stats(y_pred, d.y_train)

        print "TEST"
        y_pred = m.y.eval(feed_dict={m.x: d.x_test}, session=sess)
        report_poly_stats(y_pred, d.y_test)


def report_poly_stats(y_pred, y_gold):
    print "     |             ON              |             OFF             |"
    print "--------------------------------------------------------------------------------------------------"
    print "Note | Count       Mean    Std Dev | Count       Mean    Std Dev |     Prec     Recall         F1"
    print "--------------------------------------------------------------------------------------------------"

    for n in range(0, 12):
        y_pred_n = y_pred[:, n]
        y_true_n = y_gold[:, n]

        ons = y_pred_n[y_true_n == 1]
        offs = y_pred_n[y_true_n == 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print "%4d | %5d   %f   %f | %5d   %f   %f | %f   %f   %f" % (
                n,
                len(ons),
                float(np.mean(ons)),
                float(np.std(ons)),
                len(offs),
                float(np.mean(offs)),
                float(np.std(offs)),
                precision_score(y_true_n, y_pred_n >= 0.5),
                recall_score(y_true_n, y_pred_n >= 0.5),
                f1_score(y_true_n, y_pred_n >= 0.5)
            )

        fpr, tpr, thresholds = roc_curve(y_true_n, y_pred_n)
        plt.plot(fpr, tpr)

    fpr, tpr, thresholds = roc_curve(y_gold.flatten(), y_pred.flatten())
    plt.plot(fpr, tpr, linewidth=2)
    plt.legend(map(str, range(0, 12)) + ["overall"], loc='lower right')
    plt.show()


def run_one_hot_joint_model(p, from_cache=True):
    with tf.Session() as sess:
        d = data.load(
            p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
        ).to_one_hot().to_padded(p.padding).to_shuffled()
        m = model.feed_forward_model(
                d.features,
                p.outputs() + 1,
                hidden_nodes=p.hidden_nodes,
                loss_function="cross_entropy",
                learning_rate=p.learning_rate)
        m.set_report("ACCURACY", m.accuracy())

        sess.run(tf.initialize_all_variables())
        train_model(p.epochs, m, d, report_epochs=5)

        persist.save(sess, m, d, p)

        print "Training stats:"
        report_stats(d.x_train, d.y_train, m, sess)

        print "Testing stats:"
        report_stats(d.x_test, d.y_test, m, sess)


def report_stats(x, y, m, sess):
    y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
    gold = y.argmax(axis=1)
    pred = y_pred.argmax(axis=1)
    print "GOLD COUNTS"
    print counts(gold)
    print "PREDICTED COUNTS"
    print counts(pred)
    print confusion_matrix(gold, pred, range(13))
    plot_confusion_heat_map(gold, pred)


def show_roc_curve(x, y, m, sess):
    y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
    fpr, tpr, thresholds = roc_curve(y.flatten(), y_pred.flatten())
    plt.plot(fpr, tpr)
    plt.show()


def plot_confusion_heat_map(gold, pred):
    matrix = confusion_matrix(gold, pred, range(np.min(gold), np.max(gold)))
    print matrix.shape
    plt.pcolormesh(range(matrix.shape[0]), range(matrix.shape[0]), matrix)
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.show()


def counts(nums):
    c = defaultdict(int)
    for i in nums:
        c[i] += 1
    return c


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

    d = data.load(train_size, test_size, slice_samples, from_cache, batch_size, corpus, 1, 128).to_binary_one_hot()

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


def run_best(corpus):
    # TODO just search graphs for this
    if corpus == "two_piano_one_octave":
        run_joint_model(
            Params(
                epochs=100,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="two_piano_one_octave",
                learning_rate=0.5,
                lower=60,
                upper=72,
                padding=0
            )
        )
    elif corpus == "mono_piano_simple":
        run_one_hot_joint_model(
            Params(
                epochs=100,
                train_size=90,
                test_size=10,
                hidden_nodes=[],
                corpus="mono_piano_simple",
                learning_rate=0.05,
                lower=21,
                upper=109,
                padding=0
            )
        )
    elif corpus == "mono_piano_one_octave":
        run_one_hot_joint_model(
            Params(
                epochs=100,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="mono_piano_one_octave",
                learning_rate=0.05,
                lower=60,
                upper=72,
                padding=0
            )
        )
    elif corpus == "mono_piano_two_octaves":
        run_one_hot_joint_model(
            Params(
                epochs=100,
                train_size=40,
                test_size=10,
                hidden_nodes=[],
                corpus="mono_piano_two_octaves",
                learning_rate=0.05,
                lower=48,
                upper=72,
                padding=0
            )
        )
    else:
        assert False


if __name__ == "__main__":
    run_joint_model(
        Params(
            epochs=500,
            train_size=40,
            test_size=10,
            hidden_nodes=[],
            corpus="two_piano_one_octave",
            learning_rate=0.05,
            lower=60,
            upper=72,
            padding=0
        )
    )

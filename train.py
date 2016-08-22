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
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
from os import devnull


def train_model(epochs, m, d, report_epochs=10):
    print "Training model with %d features..." % d.features
    for j in range(epochs + 1):
        if j == epochs or j % report_epochs == 0:
            sys.stdout.write("EPOCH %03d/%d - TRAIN %s: %0.8f - TEST %s: %0.8f\n" %
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
        ).to_padded(p.padding).to_shuffled().to_sparse()
        m = model.feed_forward_model(
            d.features,
            p.outputs(),
            hidden_nodes=p.hidden_nodes,
            learning_rate=p.learning_rate,
            dropout=True
        )

        sess.run(tf.initialize_all_variables())
        train_model(p.epochs, m, d, report_epochs=1)

        persist.save(sess, m, d, p)

        print "TRAIN"
        y_pred = m.y.eval(feed_dict={m.x: d.x_train}, session=sess)
        report_poly_stats(y_pred, d.y_train)

        print "TEST"
        y_pred = m.y.eval(feed_dict={m.x: d.x_test}, session=sess)
        report_poly_stats(y_pred, d.y_test)


def run_sequence_model(p, from_cache=True):
    with tf.Session() as sess:
        d = data.load(
            p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
        ).to_padded(p.padding).to_sparse().to_sequences(p.steps)

        m = model.rnn_model(d.features, p.outputs(), p.steps, p.hidden, p.graph_type, p.learning_rate)

        sess.run(tf.initialize_all_variables())

        report_epochs = 1
        i_state_shape = p.outputs()
        i_state_shape = i_state_shape * 2 if p.graph_type == 'lstm' else i_state_shape

        for j in range(p.epochs + 1):

            if j == p.epochs or j % report_epochs == 0:
                sys.stdout.write("EPOCH %03d/%d - TRAIN %s: %0.8f - TEST %s: %0.8f\n" %
                                 (
                                     j,
                                     p.epochs,
                                     m.report_name,
                                     m.report_target.eval(feed_dict={
                                         m.x:       d.x_train,
                                         m.y_gold:  d.y_train,
                                         m.i_state: np.zeros([d.n_train, i_state_shape])
                                     }),
                                     m.report_name,
                                     m.report_target.eval(feed_dict={
                                         m.x:       d.x_test,
                                         m.y_gold:  d.y_test,
                                         m.i_state: np.zeros([d.n_test, i_state_shape])
                                     })
                                 ))
                sys.stdout.flush()

            if j < p.epochs:
                for k in range(d.batches):
                    sys.stdout.write("EPOCH %03d/%d - BATCH %03d/%d\r" % (j + 1, p.epochs, k + 1, d.batches))
                    sys.stdout.flush()

                    start = k * d.batch_size
                    stop = (k + 1) * d.batch_size

                    m.train_step.run(feed_dict={
                        m.x:       d.x_train[start:stop, :, :],
                        m.y_gold:  d.y_train[start:stop, :, :],
                        m.i_state: np.zeros([p.batch_size, i_state_shape])
                    })

        # persist.save(sess, m, d, p)

        y_pred_train = m.y.eval(feed_dict={m.x: d.x_train, m.i_state: np.zeros([d.n_train, i_state_shape])}, session=sess)
        y_pred_test = m.y.eval(feed_dict={m.x: d.x_test, m.i_state: np.zeros([d.n_test, i_state_shape])}, session=sess)

        print "TRAIN"
        report_poly_stats(squash_sequences(y_pred_train), squash_sequences(d.y_train))

        print "TEST"
        report_poly_stats(squash_sequences(y_pred_test), squash_sequences(d.y_test))


def squash_sequences(foo):
    return np.reshape(foo, [-1, foo.shape[-1]])


def report_poly_stats(y_pred, y_gold, show_graph=True):
    print "     |              ON               |              OFF              |"
    print "----------------------------------------------------------------------" \
          "------------------------------------------"
    print "Note |   Count       Mean    Std Dev |   Count       Mean    Std Dev |" \
          "     Prec     Recall         F1    ROC AUC"
    print "----------------------------------------------------------------------" \
          "------------------------------------------"

    notes = range(0, y_gold.shape[1])

    for n in notes:
        y_pred_n = y_pred[:, n]
        y_true_n = y_gold[:, n]

        ons = y_pred_n[y_true_n == 1]
        offs = y_pred_n[y_true_n == 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print "%4d | %7d   %f   %f | %7d   %f   %f | %f   %f   %f   %f" % (
                n,
                len(ons),
                float(np.mean(ons)),
                float(np.std(ons)),
                len(offs),
                float(np.mean(offs)),
                float(np.std(offs)),
                precision_score(y_true_n, y_pred_n >= 0.5),
                recall_score(y_true_n, y_pred_n >= 0.5),
                f1_score(y_true_n, y_pred_n >= 0.5),
                roc_auc_score(y_true_n, y_pred_n)
            )

        if show_graph:
            fpr, tpr, thresholds = roc_curve(y_true_n, y_pred_n)
            plt.plot(fpr, tpr)

    if show_graph:
        fpr, tpr, thresholds = roc_curve(y_gold.flatten(), y_pred.flatten())
        plt.plot(fpr, tpr, linewidth=2)
        plt.legend(map(str, notes) + ["overall"], loc='lower right')
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
                learning_rate=p.learning_rate,
                one_hot=True)
        m.set_report("ACCURACY", m.accuracy())

        sess.run(tf.initialize_all_variables())
        train_model(p.epochs, m, d, report_epochs=5)

        persist.save(sess, m, d, p)

        print "Training stats:"
        report_mono_stats(d.x_train, d.y_train, m, sess)

        print "Testing stats:"
        report_mono_stats(d.x_test, d.y_test, m, sess)


def report_mono_stats(x, y, m, sess):
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


def run_best(corpus):
    # TODO just search graphs for this
    if corpus == "two_piano_one_octave":
        run_joint_model(
            Params(
                epochs=10,
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
    elif corpus == "two_piano_one_octave_big":
        run_joint_model(
            Params(
                epochs=50,
                train_size=400,
                test_size=100,
                hidden_nodes=[],
                corpus="two_piano_one_octave_big",
                learning_rate=0.05,
                lower=60,
                upper=72,
                padding=0
            )
        )
    elif corpus == "five_piano_simple":
        run_joint_model(
            Params(
                epochs=4,
                train_size=400,
                test_size=100,
                hidden_nodes=[],
                corpus="five_piano_simple",
                learning_rate=0.1,
                lower=21,
                upper=109,
                padding=0
            )
        )
    else:
        assert False


if __name__ == "__main__":
    run_sequence_model(
        Params(
            epochs=100,
            train_size=5,
            test_size=5,
            hidden_nodes=[],
            corpus="two_piano_one_octave",
            learning_rate=0.002,
            lower=60,
            upper=72,
            padding=0,
            batch_size=1,
            steps=50,
            hidden=12,
            graph_type="rnn"
        )
    )

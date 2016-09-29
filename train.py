import data
import generate
import midi
import model
import numpy as np
import persist
import slicer
import sys
import tensorflow as tf
import time
import warnings
from collections import defaultdict
from domain import Params
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
from os import devnull
from scipy.optimize import minimize_scalar as minimize


def train_frame_model(epochs, m, d, report_epochs=10, shuffle=True, log=True, early_stop=False):
    batches, batch_size = d.batches, d.batch_size
    epoch_time = 0.0
    j_last = -1

    best_error = 10000000.0
    best_epoch = 0

    if log:
        print "Training frame model with [%d] batches of size [%d]" % (batches, batch_size)
    for j in range(epochs + 1):
        if shuffle:
            d.shuffle_frames()
        else:
            d.shuffle_sequences()
        t1 = time.time()
        if log and (j == epochs or j % report_epochs == 0):
            sys.stdout.write("EPOCH %03d/%d - DEV %s: %0.8f (%0.8f) - TEST %s: %0.8f (%0.8f) - TIME: %0.4fs\n" %
                             (
                                 j,
                                 epochs,
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.dev_labelled_feed(d)),
                                 m.loss.eval(feed_dict=m.dev_labelled_feed(d)),
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.test_labelled_feed(d)),
                                 m.loss.eval(feed_dict=m.test_labelled_feed(d)),
                                 float(epoch_time) / float(j - j_last)
                             ))
            j_last = j
            epoch_time = 0.0
            sys.stdout.flush()

        if early_stop:
            test_error = m.report_target.eval(feed_dict=m.test_labelled_feed(d))
            if test_error < best_error:
                best_epoch = j
                best_error = test_error
            elif j > best_epoch + 5:
                print "5 epochs without improvement - STOP! Best epoch was [%d] error [%f]" % (best_epoch, best_error)
                break

        if j < epochs:
            for k in range(batches):
                if log:
                    sys.stdout.write("EPOCH %03d/%d - BATCH %04d/%d\r" % (j + 1, epochs, k + 1, batches))
                    sys.stdout.flush()

                start = k * batch_size
                stop = (k + 1) * batch_size

                m.train_step.run(feed_dict=m.train_batch_feed(d, start, stop))

        t2 = time.time()
        epoch_time += (t2 - t1)


def train_sequence_model(epochs, m, d, report_epochs):
    epoch_time = 0.0
    j_last = -1
    print "Training sequence model with [%d] batches of size [%d]" % (d.batches, d.batch_size)
    for j in range(epochs + 1):
        t1 = time.time()
        d.shuffle_sequences()
        if j == epochs or j % report_epochs == 0:
            sys.stdout.write("EPOCH %03d/%d - DEV %s: %0.8f (%0.8f) - TEST %s: %0.8f (%0.8f) - TIME: %0.4fs\n" %
                             (
                                 j,
                                 epochs,
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.dev_labelled_feed(d)),
                                 0,  # m.loss.eval(feed_dict=m.dev_labelled_feed(d)),
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.test_labelled_feed(d)),
                                 0,  # m.loss.eval(feed_dict=m.test_labelled_feed(d)),
                                 float(epoch_time) / float(j - j_last)
                             ))
            j_last = j
            epoch_time = 0.0
            sys.stdout.flush()

        if j < epochs:
            for k in range(d.batches):
                sys.stdout.write("EPOCH %03d/%d - BATCH %03d/%d\r" % (j + 1, epochs, k + 1, d.batches))
                sys.stdout.flush()

                start = k * d.batch_size
                stop = (k + 1) * d.batch_size

                m.train_step.run(feed_dict={
                    m.x:       d.x_train[start:stop, :, :],
                    m.y_gold:  d.y_train[start:stop, :, :]
                })
        t2 = time.time()
        epoch_time += (t2 - t1)


def load_data(p, from_cache):
    d = data.load(
        p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
    ).to_padded(p.padding).to_sparse()
    return d


def run_frame_model(
        p,
        from_cache=True,
        d=None,
        report_epochs=1,
        pre_ps=list(),
        pre_d=None,
        ui=True,
        log=True,
        early_stop=False):

    if not d:
        d = load_data(p, from_cache)
        if p.subsample:
            d.subsample_frames(p.subsample)

    with tf.Session() as sess:

        if p.graph_type == 'ladder':
            m = model.ladder_model(
                d.features,
                p.outputs(),
                p.learning_rate,
                p.hidden_nodes,
                p.noise_var,
                p.noise_costs
            )

            # TODO: why do I need to bind the training variable here?
            sess.run(tf.initialize_all_variables(), feed_dict={m.training: True})
            sess.run(tf.initialize_all_variables(), feed_dict={m.training: False})

        elif p.graph_type == 'mlp':
            m = model.feed_forward_model(
                d.features,
                p.outputs(),
                p.learning_rate,
                p.hidden_nodes,
                p.dropout
            )
            sess.run(tf.initialize_all_variables())
        else:
            assert False, "Unknown graph type [%s]" % p.graph_type

        for pre_p in pre_ps:
            if not pre_d:
                pre_d = load_data(pre_p, from_cache)
            pre_d.set_test(d.x_test, d.y_test)
            print "Pre-training with %s" % pre_p.corpus
            train_frame_model(pre_p.epochs, m, pre_d, report_epochs, log=log)
            print "Completed pre-training"

        train_frame_model(p.epochs, m, d, report_epochs, log=log, early_stop=early_stop)

        y_pred_train = m.y.eval(feed_dict=m.train_unlabelled_feed(d), session=sess)
        y_pred_test = m.y.eval(feed_dict=m.test_unlabelled_feed(d), session=sess)

        def f1(t):
            return 1 - f1_score(d.y_test.flatten(), y_pred_test.flatten() >= t)

        threshold = minimize(f1, bounds=(0, 1), method='Bounded').x
        print "Found threshold [%f]" % threshold
        graph_id, test_error = persist.save(sess, m, d, p, threshold)

        if log:
            report_run_results(y_pred_train, d.y_train, y_pred_test, d.y_test, ui, threshold)

        return graph_id, test_error


def run_hierarchical_model(p, from_cache=True, report_epochs=1, ui=True):
    with tf.Session() as sess:
        d = load_data(p, from_cache).to_sequences(p.steps)

        if p.graph_type == 'mlp_mlp':
            frame, sequence = model.hierarchical_deep_network(
                d.features,
                p.outputs(),
                p.steps,
                p.frame_hidden_nodes,
                p.frame_dropout,
                p.frame_learning_rate,
                p.sequence_hidden_nodes,
                p.sequence_dropout,
                p.sequence_learning_rate
            )
        elif p.graph_type == 'mlp_rnn':
            frame, sequence = model.hierarchical_recurrent_network(
                d.features,
                p.outputs(),
                p.steps,
                p.frame_hidden_nodes,
                p.frame_dropout,
                p.frame_learning_rate,
                p.rnn_graph_type,
                p.sequence_learning_rate
            )
        else:
            assert False, "Unexpected graph type [%s]" % p.graph_type

        sess.run(tf.initialize_all_variables())

        def unroll_sequences(foo):
            return np.reshape(foo, [-1, foo.shape[-1]])

        print "***** Training on frames using [%s]" % p.corpus
        train_frame_model(p.frame_epochs, frame, d, report_epochs, shuffle=False)

        y_pred_train = unroll_sequences(frame.y.eval(feed_dict={sequence.x: d.x_train}, session=sess))
        y_pred_test = unroll_sequences(frame.y.eval(feed_dict={sequence.x: d.x_test}, session=sess))

        y_gold_train = unroll_sequences(d.y_train)
        y_gold_test = unroll_sequences(d.y_test)

        def f1(t):
            return 1 - f1_score(y_gold_test.flatten(), y_pred_test.flatten() >= t)

        threshold = minimize(f1, bounds=(0, 1), method='Bounded').x

        report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui, threshold)

        print "***** Training on sequences using [%s]" % p.corpus
        train_sequence_model(p.epochs, sequence, d, report_epochs)

        y_pred_train = unroll_sequences(sequence.y.eval(feed_dict={sequence.x: d.x_train}, session=sess))
        y_pred_test = unroll_sequences(sequence.y.eval(feed_dict={sequence.x: d.x_test}, session=sess))

        y_gold_train = unroll_sequences(d.y_train)
        y_gold_test = unroll_sequences(d.y_test)

        def f1(t):
            return 1 - f1_score(y_gold_test.flatten(), y_pred_test.flatten() >= t)

        threshold = minimize(f1, bounds=(0, 1), method='Bounded').x
        print "Found threshold [%f]" % threshold
        persist.save(sess, sequence, d, p, threshold)

        report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui, threshold)


def report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui, threshold):
    print "TRAIN"
    report_poly_stats(y_pred_train, y_gold_train, breakdown=False, ui=ui, threshold=threshold)

    print "TEST"
    report_poly_stats(y_pred_test, y_gold_test, breakdown=False, ui=ui, threshold=threshold)

    if ui:
        plot_piano_roll(y_pred_train[:1500, :], y_gold_train[:1500, :])
        plot_piano_roll(y_pred_test[:1500, :], y_gold_test[:1500, :])


def report_poly_stats(y_pred, y_gold, breakdown=True, ui=True, threshold=0.5):

    notes = range(0, y_gold.shape[1])

    print "     |              ON               |              OFF              |"
    print "-" * 112
    print "Note |   Count       Mean    Std Dev |   Count       Mean    Std Dev |" \
          "     Prec     Recall         F1    ROC AUC"
    print "-" * 112
    if breakdown:
        for n in notes:
            y_pred_n = y_pred[:, n]
            y_true_n = y_gold[:, n]

            ons = y_pred_n[y_true_n == 1]
            offs = y_pred_n[y_true_n == 0]

            if ons.size == 0 or offs.size == 0:
                print "x" * 112
                continue

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

            if ui:
                fpr, tpr, thresholds = roc_curve(y_true_n, y_pred_n)
                plt.plot(fpr, tpr)
        print "-" * 112
    ons = y_pred[y_gold == 1]
    offs = y_pred[y_gold == 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print "ALL  | %7d   %f   %f | %7d   %f   %f | %f   %f   %f   %f" % (
            len(ons),
            float(np.mean(ons)),
            float(np.std(ons)),
            len(offs),
            float(np.mean(offs)),
            float(np.std(offs)),
            precision_score(y_gold, y_pred >= threshold),
            recall_score(y_gold, y_pred >= threshold),
            f1_score(y_gold, y_pred >= threshold),
            roc_auc_score(y_gold, y_pred)
        )

    if ui:
        fpr, tpr, thresholds = roc_curve(y_gold.flatten(), y_pred.flatten())
        plt.plot(fpr, tpr, linewidth=2)
        if breakdown:
            plt.legend(map(str, notes) + ["overall"], loc='lower right')
        plt.show()


def run_one_hot_joint_model(p, from_cache=True):
    with tf.Session() as sess:
        d = data.load(
            p.train_size, p.test_size, p.slice_samples, from_cache, p.batch_size, p.corpus, p.lower, p.upper
        ).to_one_hot().to_padded(p.padding)
        m = model.feed_forward_model(
                d.features,
                p.outputs() + 1,
                hidden_nodes=p.hidden_nodes,
                learning_rate=p.learning_rate,
                one_hot=True)
        m.set_report("ACCURACY", m.accuracy())

        sess.run(tf.initialize_all_variables())
        train_frame_model(p.epochs, m, d, report_epochs=5)

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


def plot_piano_roll(y_pred, y_gold):

    plt.subplot(2, 1, 1)
    plt.pcolormesh(range(y_pred.shape[0]), range(y_pred.shape[1]), np.transpose(y_pred))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.pcolormesh(range(y_gold.shape[0]), range(y_gold.shape[1]), np.transpose(y_gold))
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
    ex, _ = data.load_x("output/sanity.wav", slice_samples, coarse=False)
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
    generate.write_wav_file("output/sanity_pred.mid", "output/sanity_pred_deep.wav", open(devnull, 'w'), 0)


if __name__ == "__main__":
    run_frame_model(
        Params(
            epochs=200,
            train_size=100,
            test_size=20,
            corpus="16k_piano_notes_88_poly_3_to_15_velocity_63_to_127",
            batch_size=512,
            graph_type='ladder',
            dropout=None,
            hidden_nodes=[176],
            learning_rate=0.0001,
            noise_var=0.1,
            noise_costs=[1.0, 0.1, 0.1, 0.1]
        ),
        report_epochs=1
    )


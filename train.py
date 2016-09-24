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


def train_frame_model(epochs, m, d, report_epochs=10, shuffle=True, batch_override=None):
    if batch_override:
        batches, batch_size = batch_override
    else:
        batches, batch_size = d.batches, d.batch_size
    epoch_time = 0.0
    j_last = -1
    print "Training frame model with [%d] batches of size [%d]" % (batches, batch_size)
    for j in range(epochs + 1):
        if shuffle:
            d.shuffle_frames()
        else:
            d.shuffle_sequences()
        t1 = time.time()
        if j == epochs or j % report_epochs == 0:
            sys.stdout.write("EPOCH %03d/%d - TRAIN %s: %0.8f (%0.8f) - TEST %s: %0.8f (%0.8f) - TIME: %0.4fs\n" %
                             (
                                 j,
                                 epochs,
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.train_labelled_feed(d)),
                                 m.loss.eval(feed_dict=m.train_labelled_feed(d)),
                                 m.report_name,
                                 m.report_target.eval(feed_dict=m.test_labelled_feed(d)),
                                 m.loss.eval(feed_dict=m.test_labelled_feed(d)),
                                 float(epoch_time) / float(j - j_last)
                             ))
            j_last = j
            epoch_time = 0.0
            sys.stdout.flush()

        if j < epochs:
            for k in range(batches):
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
            sys.stdout.write("EPOCH %03d/%d - TRAIN %s: %0.8f - TEST %s: %0.8f - TIME: %0.4fs\n" %
                             (
                                 j,
                                 epochs,
                                 m.report_name,
                                 m.report_target.eval(feed_dict={
                                     m.x:       d.x_train,
                                     m.y_gold:  d.y_train
                                 }),
                                 m.report_name,
                                 m.report_target.eval(feed_dict={
                                     m.x:       d.x_test,
                                     m.y_gold:  d.y_test
                                 }),
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


def run_frame_model(p, from_cache=True, d=None, report_epochs=1, pre_p=None, pre_d=None, ui=True):
    if not d:
        d = load_data(p, from_cache)
        if p.subsample:
            d.subsample_frames(p.subsample)

    with tf.Session() as sess:
        m = model.frame_model(
            p.graph_type,
            d.features,
            p.outputs(),
            p.batch_size,
            hidden_nodes=p.hidden_nodes,
            learning_rate=p.learning_rate,
            dropout=p.dropout
        )

        # TODO: why??
        sess.run(tf.initialize_all_variables(), feed_dict={m.training: True})
        sess.run(tf.initialize_all_variables(), feed_dict={m.training: False})

        if pre_p:
            if not pre_d:
                pre_d = load_data(pre_p, from_cache).shuffle_frames()
            pre_d.set_test(d.x_test, d.y_test)
            print "Pre-training with %s" % pre_p.corpus
            train_frame_model(pre_p.epochs, m, pre_d, report_epochs)
            print "Completed pre-training"

        train_frame_model(p.epochs, m, d, report_epochs)

        persist.save(sess, m, d, p)

        y_pred_train = m.y.eval(feed_dict=m.train_unlabelled_feed(d), session=sess)
        y_pred_test = m.y.eval(feed_dict=m.test_unlabelled_feed(d), session=sess)

        report_run_results(y_pred_train, d.y_train, y_pred_test, d.y_test, ui)


def run_sequence_model(p, from_cache=True, pre_p=None, report_epochs=10, d=None, pre_d=None, ui=True):
    with tf.Session() as sess:
        if not d:
            d = load_data(p, from_cache).to_sequences(p.steps)

        m = model.rnn_model(
            d.features,
            p.outputs(),
            p.steps,
            p.hidden,
            p.graph_type,
            p.learning_rate
        )

        sess.run(tf.initialize_all_variables())

        if pre_p:
            if not pre_d:
                pre_d = load_data(pre_p, from_cache).to_sequences(p.steps)
            pre_d.set_test(d.x_test, d.y_test)
            print "Pre-training with %s" % pre_p.corpus
            train_sequence_model(pre_p.epochs, m, pre_d, report_epochs)
            print "Completed pre-training"

        train_sequence_model(p.epochs, m, d, report_epochs)

        persist.save(sess, m, d, p)

        y_pred_train = unroll_sequences(m.y.eval(feed_dict={m.x: d.x_train, m.i_state: d.init_train}, session=sess))
        y_pred_test = unroll_sequences(m.y.eval(feed_dict={m.x: d.x_test, m.i_state: d.init_test}, session=sess))

        y_gold_train = unroll_sequences(d.y_train)
        y_gold_test = unroll_sequences(d.y_test)

        report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui)


def run_hybrid_model(p, ac_rate, ac_epochs, from_cache=True, pre_p=None, report_epochs=10, d=None, pre_d=None, ui=True):
    with tf.Session() as sess:

        ac, m = model.hybrid_model(
            660,
            p.outputs(),
            p.steps,
            p.hidden,
            p.hidden_nodes,
            p.graph_type,
            p.learning_rate,
            ac_rate,
            dropout=p.dropout
        )

        sess.run(tf.initialize_all_variables())

        if pre_p:
            if not pre_d:
                pre_d = load_data(pre_p, from_cache).to_sequences(p.steps)

            print "***** Pre-training on frames using [%s]" % pre_p.corpus
            train_frame_model(pre_p.epochs, ac, pre_d, report_epochs)

            print "***** Pre-training on sequences using [%s]" % pre_p.corpus
            train_sequence_model(pre_p.epochs, m, pre_d, report_epochs)

            print "***** Completed pre-training"

        if not d:
            d = load_data(p, from_cache).to_sequences(p.steps)

        print "***** Training on frames using [%s]" % p.corpus
        train_frame_model(ac_epochs, ac, d, report_epochs, shuffle=False)

        print "***** Training on sequences using [%s]" % p.corpus
        train_sequence_model(p.epochs, m, d, report_epochs)

        print "***** Completed training"

        persist.save(sess, m, d, p)

        y_pred_train = unroll_sequences(m.y.eval(feed_dict={m.x: d.x_train, m.i_state: d.init_train}, session=sess))
        y_pred_test = unroll_sequences(m.y.eval(feed_dict={m.x: d.x_test, m.i_state: d.init_test}, session=sess))

        y_gold_train = unroll_sequences(d.y_train)
        y_gold_test = unroll_sequences(d.y_test)

        report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui)


def unroll_sequences(foo):
    return np.reshape(foo.transpose(1, 0, 2), [-1, foo.shape[-1]])


def report_run_results(y_pred_train, y_gold_train, y_pred_test, y_gold_test, ui):
    print "TRAIN"
    report_poly_stats(y_pred_train, y_gold_train, breakdown=False, ui=ui)

    print "TEST"
    report_poly_stats(y_pred_test, y_gold_test, breakdown=False, ui=ui)

    if ui:
        plot_piano_roll(y_pred_test[:1500, :], y_gold_test[:1500, :])


def report_poly_stats(y_pred, y_gold, breakdown=True, ui=True):

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
            precision_score(y_gold, y_pred >= 0.5),
            recall_score(y_gold, y_pred >= 0.5),
            f1_score(y_gold, y_pred >= 0.5),
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
        ).to_one_hot().to_padded(p.padding).shuffle_frames()
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
    generate.write_wav_file("output/sanity_pred.mid", "output/sanity_pred_deep.wav", open(devnull, 'w'))


if __name__ == "__main__":

    # Scores to beat:
    # LSTM:                                     0.15517218
    # Frame: 0 hidden layers:                   0.15908915
    # Frame: 1 hidden layer:    DROPOUT: 0.5    0.15406726   (0.919213 ROC AUC)
    # Frame: 2 hidden layers:   DROPOUT: None   0.15111840   (0.923800 ROC AUC) marveled-pan's
    # Hybrid: 1 hidden layers:  DROPOUT: None   0.15685987

    run_frame_model(
        Params(
            epochs=10,
            train_size=50,
            test_size=15,
            hidden_nodes=[176, 132],
            corpus="piano_notes_88_poly_3_to_15_velocity_63_to_127",
            learning_rate=0.01,
            lower=21,
            upper=109,
            padding=0,
            batch_size=512,
            dropout=False,
            graph_type='ladder'
        ),
        report_epochs=1
    )

    # run_hybrid_model(
    #     Params(
    #         epochs=200,
    #         train_size=600,
    #         test_size=200,
    #         hidden_nodes=[176, 134],
    #         corpus="piano_notes_88_poly_3_to_15_velocity_63_to_127",
    #         learning_rate=0.001,
    #         lower=21,
    #         upper=109,
    #         padding=0,
    #         batch_size=16,
    #         steps=256,
    #         hidden=64,
    #         graph_type='lstm'
    #     ),
    #     pre_p=Params(
    #         epochs=50,
    #         train_size=48,
    #         test_size=2,
    #         hidden_nodes=[176, 132],
    #         corpus="piano_notes_88_mono_velocity_95",
    #         learning_rate=0.4,
    #         lower=21,
    #         upper=109,
    #         padding=0,
    #         steps=256,
    #         batch_size=16
    #     ),
    #     report_epochs=10,
    #     ac_rate=0.007,
    #     ac_epochs=220
    # )

    # Best 2-layer  (0.14959867 / 0.924106)  /  DROPOUT = OFF
    # run_frame_model(
    #     Params(
    #         epochs=200,
    #         train_size=600,
    #         test_size=200,
    #         hidden_nodes=[176, 132],
    #         corpus="piano_notes_88_poly_3_to_15_velocity_63_to_127",
    #         learning_rate=0.007,
    #         lower=21,
    #         upper=109,
    #         padding=0,
    #         batch_size=4096,
    #         dropout=False
    #     ),
    #     report_epochs=10,
    #     pre_p=Params(
    #         epochs=50,
    #         train_size=48,
    #         test_size=2,
    #         hidden_nodes=[176, 132],
    #         corpus="piano_notes_88_mono_velocity_95",
    #         learning_rate=0.4,
    #         lower=21,
    #         upper=109,
    #         padding=0,
    #         batch_size=4096
    #     ),
    #     ui=False
    # )

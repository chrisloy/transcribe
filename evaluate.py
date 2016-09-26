import data
import numpy as np
import os
import persist
import re
from scipy.optimize import minimize_scalar as minimize
from sklearn.metrics import f1_score
import spectrogram as sp
import tensorflow as tf


def predict(m, p, feature_file, midi_file, sess, train_flag=False):
    x, y_gold = data.load_named_pair_from_cache(feature_file, midi_file, p.lower, p.upper)
    if train_flag:
        y_pred = m.y.eval(feed_dict={m.x: x, m.training: True}, session=sess)
    else:
        y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
    return y_pred, y_gold, x


def maps_files(d='MAPS', features=False):
    in_end = '_features.p' if features else '.wav'
    return map(
        lambda x: (re.sub('\.wav$', in_end, x), re.sub('\.wav$', '.mid', x)),
        filter(
            lambda x: x.endswith('.wav') and 'MUS' in x,
            reduce(
                lambda x, y: x + y,
                (map(lambda f: os.path.join(dirpath, f), files) for dirpath, _, files in os.walk(d))
            )
        )
    )


def corpus(d='corpus/piano_notes_88_poly_3_to_15_velocity_63_to_127'):
    return map(
        lambda x: (x, re.sub('wav$', 'mid', x)),
        filter(
            lambda x: x.endswith('.wav'),
            reduce(
                lambda x, y: x + y,
                (map(lambda f: os.path.join(dirpath, f), files) for dirpath, _, files in os.walk(d))
            )
        )
    )


def test_on_maps(graph, threshold=0.15, is_ladder=False):
    maps = "MAPS_16k"
    print "Testing against [%s] with [%s]" % (maps, graph)
    with tf.Session() as sess:
        m, p = persist.load(sess, graph)
        count = 0
        f = 0
        mfs = maps_files('MAPS_16k', features=True)
        for i, (wav_file, midi_file) in enumerate(mfs):
            y_pred, y_gold, x = predict(m, p, wav_file, midi_file, sess, train_flag=is_ladder)
            slices = np.shape(x)[0]
            score = f1_score(y_gold.flatten(), y_pred.flatten() >= threshold)
            count += slices
            f += (score * slices)
            error = m.report_target.eval(feed_dict={m.x: x, m.y_gold: y_gold})

            def f1(t):
                return 1 - f1_score(y_gold.flatten(), y_pred.flatten() >= t)

            best_threshold = minimize(f1, bounds=(0, 1), method='Bounded').x

            print \
                "%03d/%d - F1 %0.8f - Error %0.8f - Total F1 %0.8f (%s) BEST THRESH: %0.4f" % \
                (i + 1, len(mfs), score, error, (f / count), wav_file, best_threshold)
        print "Overall F1: %0.8f" % (f / count)


if __name__ == '__main__':
    # test_on_maps("golder-bravade")     # linear model                               ~21
    # test_on_maps("lappish-gamostely")  # trained on multi instrument (10 epochs)    ~28
    # test_on_maps("causally-nohow")     # trained on multi instrument (20 epochs)    ~28
    # test_on_maps("lyery-estrange")     # trained on multi instrument (200 epochs, 4 subsample)    ~28
    test_on_maps("thrap-zincide")     # trained on multi instrument (2000 epochs, 2 subsample, pre both)    ~25
    # graph = sys.argv[1]
    # wav = sys.argv[2]
    # midi = sys.argv[3]
    # with tf.Session() as sess:
    #     model, params = persist.load(sess, graph)
    #     print "Graph %s successfully loaded" % graph
    #     pred, gold, _ = predict(model, params, wav, midi, sess)
    # train.plot_piano_roll(pred, gold)
    # train.report_poly_stats(pred, gold, breakdown=False, ui=False)

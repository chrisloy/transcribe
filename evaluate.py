import data
import numpy as np
import os
import persist
import re
import sys
import tensorflow as tf
import train
from scipy.optimize import minimize_scalar as minimize
from sklearn.metrics import f1_score


def predict(m, p, feature_file, target_file, sess):
    x, y_gold = data.load_named_pair_from_cache(feature_file, target_file)
    if p.steps:
        x, y_gold = data.split_by_steps(x, y_gold, p.steps, p.features, p.notes)
    if p.graph_type == 'ladder':
        y_pred = m.y.eval(feed_dict={m.x: x, m.training: False}, session=sess)
    else:
        y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
    return y_pred, y_gold, x


def maps_files(d='MAPS', features=False, targets=False):
    in_end = '_features.p' if features else '.wav'
    in_sub = '_targets\.p$' if targets else '\.mid$'
    in_ppp = '_targets.p' if targets else '.mid'
    return map(
        lambda x: (re.sub(in_sub, in_end, x), x),
        filter(
            lambda x: x.endswith(in_ppp) and 'MUS' in x,
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


def test_on_maps(graph, maps="MAPS_16k_test"):
    print "Testing against [%s] with [%s]" % (maps, graph)
    with tf.Session() as sess:
        m, p, threshold = persist.load(sess, graph)
        count = 0
        f = 0
        mfs = maps_files(maps, features=True, targets=True)
        print "Using threhold [%f]" % threshold
        for i, (wav_file, midi_file) in enumerate(mfs):
            y_pred, y_gold, x = predict(m, p, wav_file, midi_file, sess)
            slices = np.shape(x)[0]
            score = f1_score(y_gold.flatten(), y_pred.flatten() >= threshold)
            count += slices
            f += (score * slices)
            if p.graph_type == 'ladder':
                error = m.report_target.eval(feed_dict={m.x: x, m.y_gold: y_gold, m.training: False})
            else:
                error = m.report_target.eval(feed_dict={m.x: x, m.y_gold: y_gold})

            def f1(t):
                return 1 - f1_score(y_gold.flatten(), y_pred.flatten() >= t)

            best_threshold = minimize(f1, bounds=(0, 1), method='Bounded').x

            print \
                "%03d/%d - F1 %0.8f - Error %0.8f - Total F1 %0.8f (%s) BEST THRESH: %0.4f" % \
                (i + 1, len(mfs), score, error, (f / count), wav_file, best_threshold)
        print "Overall F1: %0.8f" % (f / count)


def unroll(foo):
    return np.reshape(foo, [-1, foo.shape[-1]])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        test_on_maps(sys.argv[1])
    elif len(sys.argv) == 3:
        test_on_maps(sys.argv[1], sys.argv[2])
    else:
        g = sys.argv[1]
        feats = sys.argv[2]
        midi = sys.argv[3]
        with tf.Session() as ss:
            model, params = persist.load(ss, g)
            print "Graph %s successfully loaded" % g
            pred, gold, _ = predict(model, params, feats, midi, ss)
        train.plot_piano_roll(unroll(pred), unroll(gold))
        train.report_poly_stats(unroll(pred), unroll(gold), breakdown=False, ui=False)

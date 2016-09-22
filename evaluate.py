import data
import numpy as np
import os
import persist
import re
from scipy.optimize import minimize_scalar as minimize
from sklearn.metrics import f1_score
import spectrogram as sp
import tensorflow as tf


def predict(m, p, wav_file, midi_file, sess, eng=None):
    if not eng:
        eng = sp.cqt_engine(p.slice_samples, 60)
    x, y_gold = data.load_named_pair(wav_file, midi_file, eng, p.lower, p.upper)
    y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
    return y_pred, y_gold, x


def maps_files():
    return map(
        lambda x: (x, re.sub('wav$', 'mid', x)),
        filter(
            lambda x: x.endswith('.wav') and 'MUS' in x,
            reduce(
                lambda x, y: x + y,
                (map(lambda f: os.path.join(dirpath, f), files) for dirpath, _, files in os.walk('MAPS'))
            )
        )
    )


def wav_file_ok(wav_file):
    length = len(sp.load_mono(wav_file)[0])
    result = length < 15000000
    if not result:
        print ">>>> BAD WAV: %s" % wav_file
    return result


def test_on_maps(graph, threshold=0.2):
    with tf.Session() as sess:
        m, p = persist.load(sess, graph)
        count = 0
        f = 0
        mfs = maps_files()
        eng = sp.cqt_engine(p.slice_samples, 60)
        for i, (wav_file, midi_file) in enumerate(mfs):
            if wav_file_ok(wav_file):
                y_pred, y_gold, x = predict(m, p, wav_file, midi_file, sess, eng=eng)
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
    test_on_maps('fendering-uniovular')
    # graph = sys.argv[1]
    # wav = sys.argv[2]
    # midi = sys.argv[3]
    # with tf.Session() as sess:
    #     model, params = persist.load(sess, graph)
    #     print "Graph %s successfully loaded" % graph
    #     pred, gold, _ = predict(model, params, wav, midi, sess)
    # train.plot_piano_roll(pred, gold)
    # train.report_poly_stats(pred, gold, breakdown=False, ui=False)

import data
import persist
import spectrogram as sp
import sys
import tensorflow as tf
import train


def predict(m, p, wav_file, midi_file, sess):
        eng = sp.cqt_engine(p.slice_samples, 60)
        x, y_gold = data.load_named_pair(wav_file, midi_file, eng, p.lower, p.upper)
        y_pred = m.y.eval(feed_dict={m.x: x}, session=sess)
        return y_pred, y_gold


if __name__ == '__main__':
    graph = sys.argv[1]
    wav = sys.argv[2]
    midi = sys.argv[3]
    with tf.Session() as sess:
        model, params = persist.load(sess, graph)
        print "Graph %s successfully loaded" % graph
        pred, gold = predict(model, params, wav, midi, sess)
    train.plot_piano_roll(pred, gold)
    train.report_poly_stats(pred, gold, breakdown=False, ui=True)

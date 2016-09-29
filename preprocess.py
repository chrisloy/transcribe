import cPickle
import os
import re
import sys


# TODO should this use numpy.save() instead?


def cache_features(source, target, engine):
    import spectrogram
    features = spectrogram.spectrogram_cqt(source, engine)
    with open(target, 'wb') as fp:
        cPickle.dump(features, fp)


def refresh(target):
    with open(target, 'rb') as fp:
        return cPickle.load(fp)


def cache_maps(mapsdir):
    import spectrogram
    import evaluate
    eng = spectrogram.cqt_engine(512, 60)
    print "Caching %s" % mapsdir
    for j, (wav, _) in enumerate(evaluate.maps_files(mapsdir)):
        cache_features(wav, re.sub('\.wav$', '_features.p', wav), eng)
        print "DONE: %d/270" % (j+1)


def cache_maps_midi(mapsdir):
    import evaluate
    import data
    for j, (feat, mid) in enumerate(evaluate.maps_files(mapsdir, features=True)):
        if j > 36:
            print "Caching %s" % mid
            _, s = data.load_cached_x(feat, coarse=False)
            value = data.load_y(mid, s, 21, 109)
            target = re.sub('features', 'targets', feat)
            with open(target, 'wb') as fp:
                cPickle.dump(value, fp)

if __name__ == "__main__":
    # cache_maps_midi('MAPS_16k_test')
    # cache_maps('MAPS_16k')
    corpus_name = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print "Pre-processing corpus [%s] with [%d] features per octave" % (corpus_name, num)
    corpus = "corpus/%s" % corpus_name
    corpus_length = len(filter(lambda x: x.endswith(".wav"), os.listdir(corpus)))
    import spectrogram
    e = spectrogram.cqt_engine(512, num)
    for i in range(corpus_length):
        sys.stdout.write("%d/%d\r" % (i + 1, corpus_length))
        sys.stdout.flush()
        cache_features("%s/%04d.wav" % (corpus, i), "%s/%04d_features.p" % (corpus, i), e)
    print "\nDone"

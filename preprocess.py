import cPickle
import os
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


if __name__ == "__main__":
    corpus = sys.argv[1]
    corpus_length = len(filter(lambda x: x.endswith(".wav"), os.listdir(corpus)))
    import spectrogram
    e = spectrogram.cqt_engine(512, 60)
    for i in range(corpus_length):
        sys.stdout.write("%d/%d\r" % (i + 1, corpus_length))
        sys.stdout.flush()
        cache_features("%s/%04d.wav" % (corpus, i), "%s/%04d_features.p" % (corpus, i), e)
    print "\nDone"

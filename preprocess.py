import cPickle
import spectrogram
import sys


# TODO should this use numpy.save() instead?


def cache_features(source, target, engine):
    features = spectrogram.spectrogram_cqt(source, engine)
    with open(target, 'wb') as fp:
        cPickle.dump(features, fp)


def refresh(target):
    with open(target, 'rb') as fp:
        return cPickle.load(fp)


if __name__ == "__main__":
    e = spectrogram.cqt_engine(512, 60)
    for i in range(10000):
        sys.stdout.write("%d/%d\r" % (i + 1, 10000))
        sys.stdout.flush()
        cache_features("corpus/%04d.wav" % i, "corpus/%04d_features.p" % i, e)
    print "Done"

import numpy as np
import midi
import os
import preprocess
import slicer
import spectrogram
import sys
from functools import partial
from multiprocessing import Pool


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, batches, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batches = batches
        self.batch_size = batch_size
        self.features = x_train.shape[1]

    def to_binary_one_hot(self):
        # Change y labels into one-hot vectors in two dimensions.
        y_train = np.stack([1 - self.y_train, self.y_train], axis=2)
        y_test = np.stack([1 - self.y_test, self.y_test], axis=2)
        return Data(self.x_train, y_train, self.x_test, y_test, self.batches, self.batch_size)

    def to_one_hot(self):
        # Change y labels into one-hot vector in N+1 dimensions.
        y_train = np.concatenate([self.y_train, (1 - np.max(self.y_train, axis=1)).reshape(-1, 1)], axis=1)
        y_test = np.concatenate([self.y_test, (1 - np.max(self.y_test, axis=1)).reshape(-1, 1)], axis=1)
        return Data(self.x_train, y_train, self.x_test, y_test, self.batches, self.batch_size)

    def to_note(self, n):
        return Data(self.x_train, self.y_train[:, n, :], self.x_test, self.y_test[:, n, :], self.batches,
                    self.batch_size)

    def to_padded(self, n):

        total_layers = 2 * n + 1

        def pad_x(x):
            (slices, initial_features) = np.shape(x)
            longer = np.append(x, np.zeros((total_layers, initial_features)), axis=0)
            return np.tile(longer, (1, total_layers))[n:n+slices, :]

        return Data(
            pad_x(self.x_train),
            self.y_train,
            pad_x(self.x_test),
            self.y_test,
            self.batches,
            self.batch_size
        )


def to_x_and_slices(data):
    return data.astype("float32"), data.shape[1]


def load_x(wav_file, engine):
    return to_x_and_slices(spectrogram.spectrogram_cqt(wav_file, engine))


def load_cached_x(cache_file):
    return to_x_and_slices(preprocess.refresh(cache_file))


def load_y(midi_file, slices, lower, upper):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)[lower:upper, :]


def load_pair(i, engine, corpus, lower, upper):
    xi, s = load_x("%s/%04d.wav" % (corpus, i), engine)
    yi = load_y("%s/%04d.mid" % (corpus, i), s, lower, upper)
    return xi, yi


def load_pair_from_cache(i, corpus, lower, upper):
    xi, s = load_cached_x("%s/%04d_features.p" % (corpus, i))
    yi = load_y("%s/%04d.mid" % (corpus, i), s, lower, upper)
    return xi, yi


def load_slices(a, b, slice_samples, from_cache, corpus, lower, upper):

    pool = Pool(processes=8)

    x = []
    y = []

    if from_cache:
        xys = pool.map(partial(load_pair_from_cache, corpus=corpus, lower=lower, upper=upper), range(a, b))
        for xi, yi in xys:
            x.append(xi)
            y.append(yi)

    else:
        engine = spectrogram.cqt_engine(slice_samples, 60)
        for i in range(a, b):
            sys.stdout.write("%d/%d\r" % (i - a, b - a))
            sys.stdout.flush()
            xi, yi = load_pair(i, engine, corpus, lower, upper)
            x.append(xi)
            y.append(yi)

    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)

    return np.transpose(x), np.transpose(y)


def load(train_size, test_size, slice_samples, from_cache, batch_size, corpus, lower, upper):
    file_ext = ".p" if from_cache else ".wav"
    corpus_length = len(filter(lambda x: x.endswith(file_ext), os.listdir(corpus)))
    assert train_size + test_size <= corpus_length, "Cannot produce %d examples from corpus of size %d" % (
                                                    train_size + test_size, corpus_length)
    print "Loading training set...."
    x_train, y_train = load_slices(
        0, train_size, slice_samples, from_cache, corpus, lower, upper
    )

    print "Loading testing set...."
    x_test, y_test = load_slices(
        corpus_length - test_size, corpus_length, slice_samples, from_cache, corpus, lower, upper
    )

    batches = x_train.shape[0] / batch_size
    return Data(x_train, y_train, x_test, y_test, batches, batch_size)
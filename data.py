import numpy as np
import midi
import os
import preprocess
import re
import slicer
import sys
from functools import partial
from multiprocessing import Pool
from numpy.random import RandomState
from sklearn.preprocessing import PolynomialFeatures

r = RandomState(1234567890)


class Data:

    # X: (data_points, features)
    # Y: (data_points, notes)

    def __init__(self, x_train, y_train, x_test, y_test, batches, batch_size):
        assert batches > 0, "Batch size too high, zero batches constructed!"
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batches = batches
        self.batch_size = batch_size
        self.features = x_train.shape[1]
        self.n_train = x_train.shape[0]
        self.n_test = x_test.shape[0]
        self.notes = self.y_train.shape[1]

    def subsample_frames(self, factor):
        start = r.randint(factor)
        self.x_train = self.x_train[start::factor, :]
        self.y_train = self.y_train[start::factor, :]
        self.n_train = self.x_train.shape[0]
        self.batches = self.n_train / self.batch_size
        print "New batches [%d], New n_train [%d]" % (self.batches, self.n_train)

    def set_test(self, x_test, y_test):
        assert self.features == x_test.shape[-1], "%f, %f" % (self.features, x_test.shape[-1])
        assert self.notes == y_test.shape[-1]
        self.x_test = x_test
        self.y_test = y_test
        self.n_test = x_test.shape[0]

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

        if n > 0:
            total_layers = 2 * n + 1

            def pad_x(x):
                (slices, initial_features) = np.shape(x)
                longer = np.vstack((x, np.zeros((total_layers, initial_features))))
                return np.tile(longer, (1, total_layers))[n:n+slices, :]

            return Data(
                pad_x(self.x_train),
                self.y_train,
                pad_x(self.x_test),
                self.y_test,
                self.batches,
                self.batch_size
            )
        else:
            return self

    def shuffle_frames(self):
        i = r.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[i, :]
        self.y_train = self.y_train[i, :]
        return self

    def shuffle_sequences(self):
        i = r.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[i, :, :]
        self.y_train = self.y_train[i, :, :]
        return self

    def to_sparse(self, threshold=0.01):
        self.x_train[self.x_train < threshold] = 0
        self.x_test[self.x_test < threshold] = 0
        return self

    def to_sequences_old(self, sequence_length):

        seqs = self.x_train.shape[0] / sequence_length
        keep = seqs * sequence_length
        self.x_train = np.reshape(self.x_train[:keep, :], (sequence_length, seqs, self.features)).transpose(1, 0, 2)
        self.y_train = np.reshape(self.y_train[:keep, :], (sequence_length, seqs, self.notes)).transpose(1, 0, 2)

        seqs = self.x_test.shape[0] / sequence_length
        keep = seqs * sequence_length
        self.x_test = np.reshape(self.x_test[:keep, :], (sequence_length, seqs, self.features)).transpose(1, 0, 2)
        self.y_test = np.reshape(self.y_test[:keep, :], (sequence_length, seqs, self.notes)).transpose(1, 0, 2)

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        self.batches = self.n_train / self.batch_size

        print "Separated data into [%d] train and [%d] test sequences of length [%d]" %\
              (self.n_train, self.n_test, sequence_length)

        return self

    def to_sequences(self, steps):

        self.x_train, self.y_train = split_by_steps(self.x_train, self.y_train, steps, self.features, self.notes)
        self.x_test, self.y_test = split_by_steps(self.x_test, self.y_test, steps, self.features, self.notes)

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        self.batches = self.n_train / self.batch_size

        assert self.batches > 0, "Batch size too high, zero batches constructed!"

        print "Separated data into [%d] train and [%d] test sequences of length [%d]" %\
              (self.n_train, self.n_test, steps)

        return self


def split_by_steps(x, y, steps, features, notes):
    keep = (x.shape[0] / steps) * steps
    x = np.reshape(x[:keep, :], [-1, steps, features])
    y = np.reshape(y[:keep, :], [-1, steps, notes])
    return x, y


def poly_kernel(x):
    return PolynomialFeatures(degree=2, interaction_only=True).fit_transform(x)


def to_x_and_slices(data):
    return data.astype("float32"), data.shape[1]


def load_x(wav_file, engine, coarse):
    import spectrogram
    x, s = to_x_and_slices(spectrogram.spectrogram_cqt(wav_file, engine))
    return coarsely(x, s, coarse)


def load_cached_x(cache_file, coarse):
    x, s = to_x_and_slices(preprocess.refresh(cache_file))
    return coarsely(x, s, coarse)


def load_cached_y(targets_file):
    return preprocess.refresh(targets_file)


def coarsely(x, s, coarse):
    if coarse:
        new_s = s / 3
        max_i = x.shape[1] - (x.shape[1] % new_s)
        return re_bin(x[:, :max_i], (x.shape[0], new_s)), new_s
    else:
        return x, s


def load_y(midi_file, slices, lower=21, upper=109):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)[lower:upper, :]


def load_pair(i, engine, corpus, lower, upper, coarse):
    xi, s = load_x("%s/%04d.wav" % (corpus, i), engine, coarse)
    yi = load_y("%s/%04d.mid" % (corpus, i), s, lower, upper)
    return xi, yi


def load_named_pair(wav_file, midi_file, engine, lower, upper):
    xi, s = load_x(wav_file, engine, coarse=False)
    yi = load_y(midi_file, s, lower, upper)
    return np.transpose(xi), np.transpose(yi)


def load_named_pair_from_cache(features_file, targets_file, cached_y=True):
    xi, s = load_cached_x(features_file, coarse=False)
    if cached_y:
        yi = load_cached_y(targets_file)
    else:
        yi = load_y(targets_file, s, 21, 109)
    return np.transpose(xi), np.transpose(yi)


def load_pair_from_cache(i, corpus, lower, upper, coarse):
    xi, s = load_cached_x("%s/%04d_features.p" % (corpus, i), coarse)
    yi = load_y("%s/%04d.mid" % (corpus, i), s, lower, upper)
    return xi, yi


def re_bin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_slices(a, b, slice_samples, from_cache, corpus, lower, upper, coarse):

    pool = Pool(processes=8)

    x = []
    y = []

    if from_cache:
        xys = pool.map(
            partial(load_pair_from_cache, corpus=corpus, lower=lower, upper=upper, coarse=coarse), range(a, b))

        for xi, yi in xys:
            x.append(xi)
            y.append(yi)

    else:
        import spectrogram
        engine = spectrogram.cqt_engine(slice_samples, 60)
        for i in range(a, b):
            sys.stdout.write("%d/%d\r" % (i - a, b - a))
            sys.stdout.flush()
            xi, yi = load_pair(i, engine, corpus, lower, upper, coarse)
            x.append(xi)
            y.append(yi)

    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)

    return np.transpose(x), np.transpose(y)


def load(train_size, test_size, slice_samples, from_cache, batch_size, corpus_name, lower, upper, coarse=False):
    file_ext = ".p" if from_cache else ".wav"
    corpus = "corpus/%s" % corpus_name
    corpus_length = len(filter(lambda x: x.endswith(file_ext), os.listdir(corpus)))
    print "Corpus [%s] contains [%d] tracks" % (corpus_name, corpus_length)
    assert train_size + test_size <= corpus_length, "Cannot produce %d examples from corpus of size %d" % (
                                                    train_size + test_size, corpus_length)
    print "Loading training set...."
    x_train, y_train = load_slices(
        0, train_size, slice_samples, from_cache, corpus, lower, upper, coarse
    )

    print "Loading testing set...."
    x_test, y_test = load_slices(
        corpus_length - test_size, corpus_length, slice_samples, from_cache, corpus, lower, upper, coarse
    )

    batches = x_train.shape[0] / batch_size
    print "Corpus loaded with [%d] training data and [%d] testing data" % (x_train.shape[0], x_test.shape[0])
    return Data(x_train, y_train, x_test, y_test, batches, batch_size)


def load_maps_file(mid):
    print "Loading [%s]" % mid
    return load_named_pair_from_cache(re.sub("_targets\.p$", "_features.p", mid), mid)


def load_maps_instruments(batch_size, train_files, test_files):

    pool = Pool(processes=8)

    def load_maps_files(ms):
        x = []
        y = []

        xys = pool.map(load_maps_file, ms)

        for xi, yi in xys:
            x.append(xi)
            y.append(yi)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        print "Corpus loaded with [%d] data points." % x.shape[0]
        return x, y, x.shape[0] / batch_size

    x_train, y_train, batches = load_maps_files(train_files)
    x_test, y_test, _ = load_maps_files(test_files)

    return Data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=batch_size,
        batches=batches
    )


def maps_cross_instruments(batch_size):
    train_raw = map(lambda x: "MAPS_16k/%s" % x, filter(lambda x: "targets" in x, os.listdir("MAPS_16k")))
    test_raw = map(lambda x: "MAPS_16k_test/%s" % x, filter(lambda x: "targets" in x, os.listdir("MAPS_16k_test")))
    train = filter(lambda x: "ENSTDk" not in x, train_raw + test_raw)
    test = filter(lambda x: "ENSTDk" in x, train_raw + test_raw)
    assert len(train) == 210, train
    assert len(test) == 60, test
    return load_maps_instruments(batch_size, train, test)


def maps_all_instruments(batch_size):
    train = map(lambda x: "MAPS_16k/%s" % x, filter(lambda x: "targets" in x, os.listdir("MAPS_16k")))
    test = map(lambda x: "MAPS_16k_test/%s" % x, filter(lambda x: "targets" in x, os.listdir("MAPS_16k_test")))
    return load_maps_instruments(batch_size, train, test)


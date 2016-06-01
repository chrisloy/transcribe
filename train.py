import midi
import numpy as np
import slicer
import spectrogram


def load_x(wav_file, slices):
    _, _, data = spectrogram.load_slice(wav_file, slices)
    return data[:, 0:slices]  # TODO why do I have to trim the end?


def load_y(midi_file, slices):
    m = midi.read_midifile(midi_file)
    return slicer.slice_midi_into(m, slices)


def load_slices():
    x = []
    y = []
    for i in range(500, 510):
        s = 2000
        yi = load_y("output/%04d.mid" % i, s)
        xi = load_x("output/%04d.wav" % i, s)
        assert xi.shape[1] == yi.shape[1]
        x.append(xi)
        y.append(yi)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)
    return x, y


def run():
    x, y = load_slices()

if __name__ == "__main__":
    run()
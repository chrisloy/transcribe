from scipy.io import wavfile
import scipy.signal as sig
from matplotlib import pyplot as plt
import numpy as np


def load(file_name):
    _, stereo = wavfile.read(file_name)
    mono = stereo[:, 0] + stereo[:, 1]
    fs, ts, data = sig.spectrogram(mono)
    return data


def plot(spectrogram):
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.log(spectrogram), origin="lower", aspect="auto", cmap="jet", interpolation="none")
    plt.colorbar()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    plot(load('output/0128.wav'))

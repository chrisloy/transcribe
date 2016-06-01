from scipy.io import wavfile
import scipy.signal as sig
from matplotlib import pyplot as plt
import numpy as np


def load(file_name):
    # Detects frequencies up to half the sampling rate
    # Each slice is 256 samples long, with a 32 overlap, leaving a size of 224 samples each
    # For a 13.25 second file, there are 44100 * 13.25 = 584k samples
    # This corresponds to a ~2600 slices taken, each with a length of approximately 5ms
    rate, stereo = wavfile.read(file_name)  # (584344, 2)
    mono = np.add(stereo[:, 0], stereo[:, 1]) / 2
    # F = every 172.265625
    # T = every 0.00507937 seconds (this is found from 44100 samples per second / 224 sample size)
    # S = amplitude (linear)
    return sig.spectrogram(mono, fs=rate)


def load_slice(file_name, slices):
    rate, stereo = wavfile.read(file_name)
    mono = np.add(stereo[:, 0], stereo[:, 1]) / 2
    nps = int(584344.0 / slices) + 32
    return sig.spectrogram(mono, fs=rate, nperseg=nps)


def plot(d, t, f):
    plt.pcolormesh(t, f, np.log(d))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    fs, ts, data = load_slice('output/0128.wav', 2500)
    np.set_printoptions(threshold='nan')
    print fs
    print ts
    print data.shape
    plot(data, ts, fs)

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


def load_mono(file_name):
    rate, stereo = wavfile.read(file_name)
    return np.add(stereo[:, 0], stereo[:, 1]) / 2, rate


def load_slice(file_name, slices):
    mono, rate = load_mono(file_name)
    nps = int(584344.0 / slices) + 32
    return sig.spectrogram(mono, fs=rate, nperseg=nps)


def plot(d, t, f):
    plt.pcolormesh(t, f, d)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()


def spectrogram_10hz(file_name, slice_samples):
    signal, rate = load_mono(file_name)
    slices = signal.shape[0] / slice_samples
    features = (slice_samples/2)+1 if slice_samples % 2 == 0 else (slice_samples+1)/2
    result = np.zeros([slices, features])
    for i in range(slices):
        signal_slice = signal[i*slice_samples:(i+1)*slice_samples]
        f = np.fft.rfft(signal_slice)
        print f.shape
        result[i, :] = np.abs(f)
    fs = np.arange(features) * 20
    ts = np.arange(slices) * (float(slice_samples) / rate)
    return fs, ts, np.transpose(result[:, :slice_samples] / np.max(result))


if __name__ == "__main__":
    y, x, s = spectrogram_10hz("output/sanity.wav", 4410)
    print s.shape
    plot(s, x, y)

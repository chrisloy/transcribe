from scipy.io import wavfile
from scipy import fftpack
import scipy.signal as sig
from matplotlib import pyplot as plt
import numpy as np
import scipy


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


def spectrogram_10hz(signal, rate, slice_samples):

    print signal.shape

    slices = signal.shape[0] / slice_samples
    result = np.zeros([slices, slice_samples])
    for i in range(slices):
        signal_slice = signal[i*slice_samples:(i+1)*slice_samples]
        f = scipy.fft(signal_slice)
        result[i, :] = np.abs(f)
    fs = np.arange(slice_samples) * 10 #scipy.fftpack.fftfreq(signal.size, 1.0/44100)
    ts = np.arange(slices) * (float(slice_samples) / rate)

    print fs.shape
    print ts.shape
    print result.shape

    # norm = np.linalg.norm(result)
    norm = np.max(result)

    # result = 20*scipy.log10(result)

    return fs, ts, np.transpose(result/norm)


if __name__ == "__main__":
    r, st = wavfile.read("output/sanity.wav")
    mono = np.add(st[:, 0], st[:, 1]) / 2
    fs, ts, s = spectrogram_10hz(mono, r, 4410)
    plot(s, ts, fs)

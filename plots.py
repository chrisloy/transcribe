import data
import evaluate as ev
import generate as gen
import math
import numpy as np
import os
import persist
import spectrogram as sp
import tensorflow as tf
from matplotlib import pyplot as plt


def sampling():
    sample = 200
    x = np.arange(sample)
    f = 2 * np.pi * x / sample
    y = np.sin(4.3 * f) + np.sin(0.7 * f) + np.cos(1.3 * f) + np.sin(f * 3.67) + np.sin(f * 3.5)

    def keep_some(tup):
        i, z = tup
        if i % 6 == 0:
            return z
        else:
            return None

    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('amplitude')

    y -= np.min(y)
    y *= math.pow(2, 16) / np.max(y)

    samples = filter(None, map(keep_some, enumerate(y)))

    plt.subplot(212)
    plt.bar(np.arange(len(samples)), samples)
    plt.xlabel('samples')
    plt.ylabel('value')

    plt.tight_layout()

    plt.savefig('figures/sampling.eps', format='eps')


def spectrogram():

    wav = 'corpus/five_piano_two_middle_octaves/0046.wav'
    mid = 'corpus/five_piano_two_middle_octaves/0046.mid'

    eng = sp.cqt_engine(512, 60)

    _, sl = data.load_x(wav, eng, coarse=False)
    _, _, fft = sp.spectrogram_10hz(wav, 512)

    midi = data.load_y(mid, sl, 40, 100)[:, 0:1000]
    cqt = sp.spectrogram_cqt(wav, eng)[290:490, 0:1000]
    fft = fft[0:50, 0:1000]

    plt.subplot(311)
    plt.pcolormesh(range(1000), range(40, 100), midi, cmap='hot', rasterized=True)
    plt.ylabel('MIDI note')

    plt.subplot(312)
    plt.pcolormesh(fft.astype("float32"), cmap='hot', rasterized=True)
    plt.ylabel('Frequency bin')

    plt.subplot(313)
    plt.pcolormesh(cqt.astype("float32"), cmap='hot', rasterized=True)
    plt.ylabel('Pitch bin')

    plt.tight_layout(pad=0.1)

    plt.savefig('figures/spectrogram.eps', format='eps')


def compare_to_real():

    eng = sp.cqt_engine(512, 60)

    close_file = "MAPS/ENSTDkCl/ISOL/CH/MAPS_ISOL_CH0.1_M_ENSTDkCl.wav"
    ambient_file = "MAPS/ENSTDkAm/ISOL/CH/MAPS_ISOL_CH0.1_M_ENSTDkAm.wav"
    midi_file = "MAPS/ENSTDkAm/ISOL/CH/MAPS_ISOL_CH0.1_M_ENSTDkAm.mid"
    gen_file = "/tmp/MAPS_ISOL_CH0.1_M_ENSTDkAm.wav"

    gen.write_wav_file(midi_file, gen_file, open(os.devnull, 'w'))

    close_sig = sp.spectrogram_cqt(close_file, eng)
    ambient_sig = sp.spectrogram_cqt(ambient_file, eng)
    fake_sig = sp.spectrogram_cqt(gen_file, eng)

    with tf.Session() as sess:
        model, params = persist.load(sess, 'fendering-uniovular')
        ambient, midi = ev.predict(model, params, ambient_file, midi_file, sess)
        close, _ = ev.predict(model, params, close_file, midi_file, sess)
        fake, _ = ev.predict(model, params, gen_file, midi_file, sess)

    # plt.subplot(3, 2, 1).axis('off')
    # plt.pcolormesh(np.transpose(midi), cmap='hot', rasterized=True)

    plt.subplot(3, 2, 1).axis('off')
    plt.pcolormesh(fake_sig[260:660, :], cmap='hot', rasterized=True)

    plt.subplot(3, 2, 2).axis('off')
    plt.pcolormesh(np.transpose(fake), cmap='hot', rasterized=True)

    plt.subplot(3, 2, 3).axis('off')
    plt.pcolormesh(close_sig[260:660, :], cmap='hot', rasterized=True)

    plt.subplot(3, 2, 4).axis('off')
    plt.pcolormesh(np.transpose(close), cmap='hot', rasterized=True)

    plt.subplot(3, 2, 5).axis('off')
    plt.pcolormesh(ambient_sig[260:660, :], cmap='hot', rasterized=True)

    plt.subplot(3, 2, 6).axis('off')
    plt.pcolormesh(np.transpose(ambient), cmap='hot', rasterized=True)

    plt.tight_layout(pad=0.0)

    plt.savefig('figures/comparison.eps', format='eps')


def frequencies():
    sample = 200
    x = np.arange(sample)
    f = 2 * np.pi * x / sample
    y1 = np.sin(4.3 * f) * 0.1
    y2 = np.sin(2.3 * f) * 0.45
    y3 = np.sin(8.3 * f) * 0.2

    plt.subplot(211)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)

    plt.subplot(212)
    plt.plot(x, y1 + y2 + y3)

    plt.tight_layout()

    plt.savefig('figures/frequencies.eps', format='eps')


if __name__ == '__main__':
    plt.figure(facecolor="white")
    # sampling()
    # spectrogram()
    # compare_to_real()
    frequencies()

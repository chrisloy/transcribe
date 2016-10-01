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


def generated_pieces():

    a = 'corpus/16k_piano_notes_88_mono_velocity_95/0046'
    b = 'corpus/16k_piano_notes_88_poly_3_to_15_velocity_63_to_127/0041'
    c = 'corpus/16k_multi_instrument_notes_88_poly_3_to_15_velocity_63_to_127/0042'

    eng = sp.cqt_engine(512, 60)

    a_spec, a_sl = data.load_x(a + '.wav', eng, coarse=False)
    a_midi = data.load_y(a + '.mid', a_sl, 21, 109)
    b_spec, b_sl = data.load_x(b + '.wav', eng, coarse=False)
    b_midi = data.load_y(b + '.mid', b_sl, 21, 109)
    c_spec, c_sl = data.load_x(c + '.wav', eng, coarse=False)
    c_midi = data.load_y(c + '.mid', c_sl, 21, 109)

    plt.subplot(321).axis('off')
    plt.pcolormesh(a_midi, cmap='hot', rasterized=True)

    plt.subplot(322).axis('off')
    plt.pcolormesh(a_spec, cmap='hot', rasterized=True)

    plt.subplot(323).axis('off')
    plt.pcolormesh(b_midi, cmap='hot', rasterized=True)

    plt.subplot(324).axis('off')
    plt.pcolormesh(b_spec, cmap='hot', rasterized=True)

    plt.subplot(325).axis('off')
    plt.pcolormesh(c_midi, cmap='hot', rasterized=True)

    plt.subplot(326).axis('off')
    plt.pcolormesh(c_spec, cmap='hot', rasterized=True)

    plt.tight_layout(pad=0.0)

    plt.savefig('figures/generated.eps', format='eps')


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
        ambient, midi, _ = ev.predict(model, params, ambient_file, midi_file, sess)
        close, _, _ = ev.predict(model, params, close_file, midi_file, sess)
        fake, _, _ = ev.predict(model, params, gen_file, midi_file, sess)

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


def curriculum():

    without_train = [
        5.82593393,
        0.41063684,
        0.32424620,
        0.29250422,
        0.27334702,
        0.26051807,
        0.25267422,
        0.24655978,
        0.23976675,
        0.23355988,
        0.22987524,
        0.22367969,
        0.22176926,
        0.21823487,
        0.21464600,
        0.21089238,
        0.20712291,
        0.20482257,
        0.19946106,
        0.19554700,
        0.19358328,
        0.19169576,
        0.18713279,
        0.18468912,
        0.18302096,
        0.17960331,
        0.17777096,
        0.17793003,
        0.17604396,
        0.17360722,
        0.17112502,
        0.17049831,
        0.16623594,
        0.16495894,
        0.16530244,
        0.16327772,
        0.16103576,
        0.16052938,
        0.15917887,
        0.15768625,
        0.15724145,
        0.15631737,
        0.15539461,
        0.15441868,
        0.15455793,
        0.15304768,
        0.15285234,
        0.15054305,
        0.15025277,
        0.14929880,
        0.14866731,
        0.14699550,
        0.14707723,
        0.14583865,
        0.14498129,
        0.14491551,
        0.14402008,
        0.14367658,
        0.14429186,
        0.14378601,
        0.14252053,
        0.14186426,
        0.14091496,
        0.14138554,
        0.14062931,
        0.14040633,
        0.14118682,
        0.14020985,
        0.13983910,
        0.13864456,
        0.13950512,
        0.13881552,
        0.13836275,
        0.13687539,
        0.13847603,
        0.13774526,
        0.13740090,
        0.13596763,
        0.13650946,
        0.13635372,
        0.13801044,
        0.13756154,
        0.13800620,
        0.13573782,
        0.13549580,
        0.13585785,
        0.13555892,
        0.13503036,
        0.13493085,
        0.13536273,
        0.13339455,
        0.13339637,
        0.13417721,
        0.13406067,
        0.13326544,
        0.13302733,
        0.13326664,
        0.13347290,
        0.13383046,
        0.13469024,
        0.13414155
    ]

    without_val = [
        5.50625849,
        0.41341099,
        0.32650238,
        0.29560792,
        0.27846959,
        0.26635307,
        0.25776473,
        0.25183928,
        0.24722864,
        0.24343489,
        0.24039565,
        0.23770517,
        0.23516323,
        0.23266599,
        0.23025896,
        0.22773071,
        0.22524419,
        0.22272350,
        0.22044465,
        0.21825184,
        0.21617877,
        0.21416365,
        0.21254836,
        0.21123365,
        0.20977440,
        0.20867726,
        0.20755678,
        0.20674221,
        0.20527077,
        0.20404005,
        0.20178339,
        0.20030430,
        0.19910611,
        0.19777711,
        0.19704549,
        0.19662890,
        0.19634470,
        0.19530524,
        0.19505255,
        0.19421153,
        0.19340758,
        0.19351295,
        0.19258606,
        0.19282600,
        0.19204122,
        0.19126789,
        0.19092996,
        0.19018327,
        0.19000936,
        0.18988335,
        0.18915610,
        0.18863547,
        0.18842006,
        0.18764944,
        0.18790402,
        0.18715957,
        0.18682127,
        0.18636324,
        0.18583767,
        0.18604828,
        0.18596068,
        0.18533352,
        0.18541047,
        0.18535200,
        0.18473953,
        0.18532768,
        0.18470636,
        0.18502302,
        0.18475206,
        0.18406177,
        0.18431607,
        0.18427145,
        0.18397200,
        0.18386328,
        0.18399248,
        0.18385792,
        0.18366766,
        0.18365771,
        0.18331644,
        0.18301356,
        0.18318892,
        0.18328536,
        0.18337603,
        0.18303132,
        0.18268819,
        0.18311149,
        0.18305530,
        0.18283728,
        0.18345678,
        0.18259069,
        0.18265504,
        0.18248153,
        0.18320005,
        0.18245021,
        0.18205164,
        0.18211824,
        0.18262395,
        0.18212190,
        0.18206097,
        0.18227470,
        0.18206416
    ]

    with_train = [
        0.58135873,
        0.34229597,
        0.30443016,
        0.28567955,
        0.27044299,
        0.25964656,
        0.25087091,
        0.24250904,
        0.23644908,
        0.23164132,
        0.22629632,
        0.22403482,
        0.21864443,
        0.21468709,
        0.21162607,
        0.20751253,
        0.20314726,
        0.19779766,
        0.19529597,
        0.19222473,
        0.19008614,
        0.18645419,
        0.18558480,
        0.18261383,
        0.17982315,
        0.17930475,
        0.17809503,
        0.17635550,
        0.17429440,
        0.17252743,
        0.17136988,
        0.17031515,
        0.16761723,
        0.16677465,
        0.16474412,
        0.16493528,
        0.16243015,
        0.16180943,
        0.16084291,
        0.16010337,
        0.15809195,
        0.15809613,
        0.15715785,
        0.15546480,
        0.15490538,
        0.15192324,
        0.15119763,
        0.15119457,
        0.14978965,
        0.15011127,
        0.14897029,
        0.14836471,
        0.14808251,
        0.14645018,
        0.14554404,
        0.14514589,
        0.14504562,
        0.14330857,
        0.14227203,
        0.14295688,
        0.14223285,
        0.14278318,
        0.14040963,
        0.14147834,
        0.14070038,
        0.14023793,
        0.13994193,
        0.13884702,
        0.13929802,
        0.13880245,
        0.13836077,
        0.13767159,
        0.13798065,
        0.13786077,
        0.13709259,
        0.13670202,
        0.13586217,
        0.13711961,
        0.13719416,
        0.13580538,
        0.13565165,
        0.13607337,
        0.13484341,
        0.13612670,
        0.13508685,
        0.13585263,
        0.13489145,
        0.13630292,
        0.13447934,
        0.13488710,
        0.13472594,
        0.13350360,
        0.13445117,
        0.13390322,
        0.13411200,
        0.13289034,
        0.13489531,
        0.13426222,
        0.13331467,
        0.13441013,
        0.13192500
    ]

    with_val = [
        0.54633951,
        0.34163311,
        0.30389169,
        0.28525168,
        0.27252865,
        0.26330870,
        0.25604233,
        0.25005099,
        0.24544415,
        0.24161361,
        0.23816115,
        0.23510635,
        0.23243937,
        0.22982810,
        0.22715853,
        0.22480299,
        0.22203392,
        0.21972960,
        0.21775283,
        0.21576948,
        0.21408756,
        0.21266265,
        0.21138382,
        0.21003723,
        0.20881979,
        0.20776781,
        0.20682888,
        0.20634025,
        0.20531993,
        0.20411158,
        0.20320344,
        0.20158637,
        0.20055917,
        0.19955917,
        0.19883688,
        0.19811207,
        0.19731395,
        0.19636062,
        0.19523206,
        0.19486150,
        0.19411592,
        0.19362502,
        0.19290507,
        0.19195193,
        0.19159482,
        0.19032976,
        0.18981789,
        0.18946689,
        0.18862084,
        0.18797404,
        0.18781078,
        0.18746090,
        0.18685840,
        0.18684399,
        0.18698741,
        0.18604308,
        0.18616253,
        0.18668559,
        0.18551998,
        0.18559487,
        0.18525356,
        0.18500690,
        0.18497056,
        0.18469794,
        0.18457337,
        0.18433981,
        0.18433556,
        0.18384194,
        0.18382779,
        0.18375100,
        0.18379760,
        0.18329051,
        0.18301879,
        0.18304688,
        0.18278052,
        0.18278474,
        0.18235436,
        0.18250701,
        0.18222165,
        0.18190533,
        0.18191318,
        0.18178742,
        0.18183014,
        0.18203995,
        0.18144175,
        0.18163919,
        0.18117690,
        0.18103431,
        0.18154252,
        0.18107046,
        0.18103848,
        0.18084908,
        0.18093336,
        0.18106295,
        0.18095122,
        0.18086772,
        0.18063727,
        0.18053024,
        0.18045039,
        0.18070751,
        0.18054177
    ]

    x = range(101)

    plt.plot(x, without_val)
    plt.plot(x, with_val, color='red')
    plt.ylim([0.18, 0.19])
    plt.xlim([40, 100])
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(["Without Curriculum", "With Curriculum"])

    plt.savefig('figures/curriculum.eps', format='eps')


if __name__ == '__main__':
    plt.figure(facecolor="white")
    # sampling()
    # spectrogram()
    # compare_to_real()
    # frequencies()
    # generated_pieces()
    curriculum()

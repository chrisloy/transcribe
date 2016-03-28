import midi
import numpy as np


def length_of_track(track):
    return sum(map(lambda x: x.tick, track))


def slice_midi(data, slice_size):
    track = data[0]
    raw = track_to_boolean_table(track)
    num = length_of_track(track) / slice_size
    slices = np.split(raw, num, axis=1)

    result = np.zeros((128, num))

    for i in range(0, num):
        result[:, i] = np.mean(slices[i], axis=1)

    return result


def track_to_boolean_table(track):
    result = np.zeros((128, length_of_track(track)))

    last = 0
    current = np.zeros(128)

    for event in track:
        up_to = last + event.tick
        for i in range(last, up_to):
            result[:, i] = current
        if type(event) == midi.events.NoteOffEvent:
            current[event.get_pitch()] = 0
        elif type(event) == midi.events.NoteOnEvent:
            current[event.get_pitch()] = 1
        else:
            pass
        last = up_to

    return result


if __name__ == "__main__":

    print "TEST: track_to_boolean_table"
    # input
    t = midi.Track([
      midi.NoteOnEvent(tick=0, pitch=5),
      midi.NoteOffEvent(tick=2, pitch=5),
      midi.NoteOnEvent(tick=4, pitch=8),
      midi.NoteOffEvent(tick=1, pitch=8),
      midi.EndOfTrackEvent(tick=1)
    ])
    # expected
    expected = np.zeros((128, 8))
    expected[5, 0:2] = 1
    expected[8, 6] = 1
    # test
    actual = track_to_boolean_table(t)
    assert np.equal(actual, expected).all()

    print "TEST: slice_midi"
    # input
    p = midi.Pattern([t])
    # expected
    expected = np.zeros((128, 2))
    expected[5, 0] = 0.5
    expected[8, 1] = 0.25
    # test
    actual = slice_midi(p, 4)
    assert np.equal(actual, expected).all()

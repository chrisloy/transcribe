import midi
import numpy as np

PITCHES = 128


def length_of_track(track):
    return sum(map(lambda x: x.tick, track))


def slice_midi(data, slice_size):
    track = data[0]  # TODO multiple tracks?
    raw = track_to_boolean_table(track)
    length = length_of_track(track)
    padding = (slice_size - (length % slice_size)) % slice_size
    num = (length + padding) / slice_size
    raw = np.append(raw, np.zeros((PITCHES, padding)), axis=1)
    slices = np.split(raw, num, axis=1)
    result = np.zeros((PITCHES, num))
    for i in range(0, num):
        result[:, i] = np.max(slices[i], axis=1)
    return result


def slice_midi_into(data, num_slices):
    track = data[0]  # TODO multiple tracks?
    raw = track_to_boolean_table(track)
    l = length_of_track(track)
    padding = ((l / num_slices) + 1) * num_slices - l
    raw = np.append(raw, np.zeros((PITCHES, padding)), axis=1)
    slices = np.split(raw, num_slices, axis=1)
    result = np.zeros((PITCHES, num_slices))
    for i in range(0, num_slices):
        result[:, i] = np.max(slices[i], axis=1)
    return result


def track_to_boolean_table(track):
    result = np.zeros((PITCHES, length_of_track(track)))

    last = 0
    current = np.zeros(PITCHES)

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


def boolean_table_to_track(table):
    result = midi.Track()

    last = np.zeros(PITCHES)
    last_event = 0

    length = table.shape[1]

    for i in range(0, length):
        this = np.round(table[:, i])
        for n in range(0, PITCHES):
            if last[n] < this[n]:
                result.append(midi.NoteOnEvent(tick=(i-last_event), pitch=n))
                last_event = i
            elif last[n] > this[n]:
                result.append(midi.NoteOffEvent(tick=(i-last_event), pitch=n))
                last_event = i
            else:
                pass
        last = this

    result.append(midi.EndOfTrackEvent(tick=(length-last_event)))

    return result


if __name__ == "__main__":

    print "TEST: track_to_boolean_table"
    # input
    t = midi.Track([
      midi.NoteOnEvent(tick=0, pitch=5),
      midi.NoteOffEvent(tick=2, pitch=5),
      midi.NoteOnEvent(tick=4, pitch=8),
      midi.NoteOffEvent(tick=1, pitch=8),
      midi.EndOfTrackEvent(tick=2)
    ])
    # expected
    expected = np.zeros((PITCHES, 9))
    expected[5, 0:2] = 1
    expected[8, 6] = 1
    # test
    actual = track_to_boolean_table(t)
    assert np.equal(actual, expected).all()

    print "TEST: slice_midi"
    # input
    p = midi.Pattern([t])
    # expected
    expected = np.zeros((PITCHES, 3))
    expected[5, 0] = 1
    expected[8, 1] = 1
    # test
    actual = slice_midi(p, 4)
    assert np.equal(actual, expected).all()

    print "TEST: boolean_table_to_track"
    test = boolean_table_to_track(track_to_boolean_table(t))
    assert test == t, test


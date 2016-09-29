import os
import midi
import random
from subprocess import call


random.seed(171253)


class Note(object):

    def __init__(self, start, stop, pitch, velocity):
        self.start = start
        self.stop = stop
        self.pitch = pitch
        self.velocity = velocity

    def length(self, measure_length):
        return (self.stop - self.start) / measure_length


PIANO_NOTES = range(21, 109)


def random_key(notes):
    scale = random.choice([
        [0, 2, 4, 5, 7, 9, 11],  # Major
        [0, 2, 4, 5, 7, 9, 10],  # Major 7th
        [0, 2, 3, 5, 7, 8, 11],  # Minor
        [0, 2, 3, 5, 7, 8, 10]   # Minor 7th
    ])
    offset = random.randint(0, 11)
    ns = map(lambda x: (x + offset) % 12, scale)
    return filter(lambda x: x % 12 in ns, notes)


def pairs(l):
    x = []
    for i in range(0, len(l), 2):
        x.append(l[i:i+2])
    return x


def up_one(note, key):
    if note == key[-1]:
        return key[0]
    else:
        return key[key.index(note) + 1]


def down_one(note, key):
    if note == key[0]:
        return key[-1]
    else:
        return key[key.index(note) - 1]


def random_velocity(lower=0, upper=127):
    return lambda: random.randint(lower, upper)


def fixed_velocity(l):
    return lambda: l


def random_polyphony(lower=10, upper=20):
    return lambda: (random.randint(lower, upper), lower, upper)


def fixed_polyphony(l):
    return lambda: (l, l, l)


def random_instrument():
    ls = range(10) + range(13, 17) + [19, 21] + range(23, 31) + range(32, 39)
    return random.choice(ls)


def fixed_instrument(l=0):
    return lambda: l


def tracking_melody(measures, measure_length, key, lower, upper, rest_probability, velocity=random_velocity):
    notes = random.randint(measures/upper, measures/lower)
    events = range(1, measures+1)
    random.shuffle(events)
    events = sorted(events[0:2*notes])
    start_stops = pairs(map(lambda e: e * measure_length, events))
    start, stop = start_stops[0]
    melody = [Note(0, stop, random.choice(key), velocity())]
    for start, stop in start_stops[1:]:
        if random.random() > rest_probability:
            last_note = melody[-1].pitch
            last_stop = melody[-1].stop
            note = random.choice([last_note, up_one(last_note, key), down_one(last_note, key)])
            melody.append(Note(last_stop, stop, note, velocity()))
    return melody


def note_to_event_pair(note):
    on = midi.NoteOnEvent(tick=note.start, velocity=note.velocity, pitch=note.pitch)
    off = midi.NoteOffEvent(tick=note.stop, pitch=note.pitch)
    return [on, off]


def random_track(measures, measure_length, polyphony, velocity, rest_probability, notes):
    track = midi.Track()
    key = random_key(notes)
    ns = []
    melodies, lower, upper = polyphony()
    for x in range(1, melodies+1):
        ns += tracking_melody(measures, measure_length, key, lower, upper, rest_probability, velocity=velocity)
    events = reduce(lambda a, b: a + b, map(note_to_event_pair, ns))
    events = sorted(events, lambda a, b: 1 if b.tick < a.tick else 0 if b.tick == a.tick else -1)
    last_tick = 0
    for event in events:
        new_tick = event.tick
        event.tick = new_tick - last_tick
        last_tick = new_tick
        track.append(event)
    return track


def random_pattern(polyphony, velocity, notes, rest_probability=0.1):
    pattern = midi.Pattern()
    pattern.append(random_track(100, 50, polyphony, velocity, rest_probability, notes))
    eot = midi.EndOfTrackEvent(tick=1)
    pattern[0].append(eot)
    return pattern


def write_wav_file(mid_file_name, wav_file_name, out_file, instrument):
    call([
        "/usr/local/bin/timidity", mid_file_name,
        "-Ow",
        "-Ei%d" % instrument(),
        "-o", wav_file_name,
        "-s", "16k"], stdout=out_file, stderr=out_file)


def generate_pair(num, out_file, corpus, polyphony, velocity, instrument, notes=PIANO_NOTES):
    mid_file_name = "%s/%04d.mid" % (corpus, num)
    wav_file_name = "%s/%04d.wav" % (corpus, num)
    midi.write_midifile(mid_file_name, random_pattern(polyphony, velocity, notes))
    write_wav_file(mid_file_name, wav_file_name, out_file, instrument)


if __name__ == "__main__":
    of = open(os.devnull, 'w')
    number = 5000
    cn = "16k_multi_instrument_notes_88_poly_3_to_15_velocity_63_to_127"
    c = "corpus/%s" % cn
    if not os.path.exists(c):
        os.makedirs(c)
        print "Created directory %s" % c
    p = random_polyphony(3, 15)
    v = random_velocity(63, 127)
    i = random_instrument
    for n in range(0, number):
        generate_pair(n, of, c, p, v, i, notes=PIANO_NOTES)
        print "Completed %d of %d in [%s]" % (n + 1, number, c)
    # import evaluate as e
    # maps = e.maps_files("MAPS_synth")
    # print "MAPS", len(maps)
    # for wav, mid in maps:
    #     write_wav_file(mid, wav, of, fixed_instrument())

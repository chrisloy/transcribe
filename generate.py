import midi
import random


class Note(object):

    def __init__(self, start, stop, pitch, velocity):
        self.start = start
        self.stop = stop
        self.pitch = pitch
        self.velocity = velocity

    def length(self, measure_length):
        return (self.stop - self.start) / measure_length


def random_key():
    scale = random.choice([
        [0, 2, 4, 5, 7, 9, 11],  # Major
        [0, 2, 4, 5, 7, 9, 10],  # Major 7th
        [0, 2, 3, 5, 7, 8, 11],  # Minor
        [0, 2, 3, 5, 7, 8, 10]   # Minor 7th
    ])
    offset = random.randint(0, 11)
    notes = map(lambda x: (x + offset) % 12, scale)
    return filter(lambda x: x % 12 in notes, range(0, 127))


def pairs(l):
    x = []
    for i in range(0, len(l), 2):
        x.append(l[i:i+2])
    return x


def shift(note, key, amount):
    shifted = (note + amount) % 127
    return shifted if note in key else shift(shifted, key, amount)


def tracking_melody(measures, measure_length, key):
    notes = random.randint(measures/20, measures/10)
    events = range(1, measures+1)
    random.shuffle(events)
    events = sorted(events[0:2*notes])
    start_stops = pairs(map(lambda e: e * measure_length, events))
    start, stop = start_stops[0]
    melody = [Note(start, stop, random.choice(key), random.randint(0, 127))]
    for start, stop in start_stops[1:]:
        last_note = melody[-1].pitch
        note = random.choice([last_note, shift(last_note, key, 1), shift(last_note, key, -1)])
        melody.append(Note(start, stop, note, random.randint(0, 127)))
    return melody


def near(last_x, xs):
    last_i = xs.index(last_x)
    new_i = min(max(last_i + int(random.gauss(0, 2)), len(xs)-1), 0)
    return xs[new_i]


def natural_melody(measures, measure_length, key):
    initial_length = random.choice([1, 2, 3]) * measure_length
    notes = [Note(0, initial_length, random.choice(key), random.randint(0, 127))]
    while True:
        length_so_far = sum(map(lambda n: n.length(measure_length), notes))
        print length_so_far
        if length_so_far > measures:
            return notes[0:-1]
        last = notes[-1]
        note = Note(
            last.stop,
            near(last.length(measure_length), [1, 2, 3]) * measure_length + last.stop,
            near(last.pitch, key),
            near(last.velocity, range(0, 127))
        )
        notes.append(note)


def note_to_event_pair(note):
    on = midi.NoteOnEvent(tick=note.start, velocity=note.velocity, pitch=note.pitch)
    off = midi.NoteOffEvent(tick=note.stop, pitch=note.pitch)
    return [on, off]


def random_track(measures, measure_length):
    track = midi.Track()
    key = random_key()
    notes = []
    for x in range(1, random.randint(10, 20)):
        notes += tracking_melody(measures, measure_length, key)
    events = reduce(lambda a, b: a + b, map(note_to_event_pair, notes))
    events = sorted(events, lambda a, b: 1 if b.tick < a.tick else -1)
    last_tick = 0
    for event in events:
        new_tick = event.tick
        event.tick = new_tick - last_tick
        last_tick = new_tick
        track.append(event)
    return track


def random_pattern():
    pattern = midi.Pattern()
    pattern.append(random_track(1000, 50))
    eot = midi.EndOfTrackEvent(tick=1)
    pattern[0].append(eot)
    return pattern


if __name__ == "__main__":
    midi.write_midifile("output/example.mid", random_pattern())

from midi import write_midifile
from generate import random_pattern


def generate_pair(id):
    file_name = "output/%d.mid" % id
    write_midifile(file_name, random_pattern())

if __name__ == "__main__":
    generate_pair(1234)

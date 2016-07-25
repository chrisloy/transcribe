import random


def new_name():
    with open("/usr/share/dict/words") as f:
        words = filter(lambda w: 4 <= len(w) <= 10, f.readlines())
        return "%s-%s" % (random.choice(words).strip().lower(), random.choice(words).strip().lower())

import random


def new_name():
    with open("/usr/share/dict/words") as f:
        words = f.readlines()
        print len(words)
        words = filter(lambda w: 4 <= len(w) <= 10, words)
        print len(words)
        return "%s-%s" % (random.choice(words).strip().lower(), random.choice(words).strip().lower())

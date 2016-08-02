import random


r = random.Random()


def new_name():
    with open("/usr/share/dict/words") as f:
        words = filter(lambda w: 4 <= len(w) <= 10, f.readlines())
        return "%s-%s" % (r.choice(words).strip().lower(), r.choice(words).strip().lower())

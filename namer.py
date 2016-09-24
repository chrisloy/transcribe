import random
import re


r = random.Random()


def clean(word):
    return re.sub(r'[^a-zA-Z0-9]','', word).strip().lower()


def new_name():
    with open("/usr/share/dict/words") as f:
        words = filter(lambda w: 4 <= len(w) <= 10, f.readlines())
        return "%s-%s" % (clean(r.choice(words)), clean(r.choice(words)))

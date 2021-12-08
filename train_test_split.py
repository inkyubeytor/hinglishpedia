import random

from load_dict import load_rom_dev, NOENG_FILE


TEST_SIZE = 1000

def load_train():
    with open(NOENG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.split('\t')] for l in lines]
    random.seed(0)
    random.shuffle(pairs)
    return pairs[:-TEST_SIZE]


def load_test():
    with open(NOENG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.split('\t')] for l in lines]
    random.seed(0)
    random.shuffle(pairs)
    return pairs[-TEST_SIZE:]


def load_all():
    with open(NOENG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.split('\t')] for l in lines]
    random.seed(0)
    random.shuffle(pairs)
    return pairs

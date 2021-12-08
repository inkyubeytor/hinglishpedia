from collections import defaultdict


DICT_FILE = "data_transliterated/rom_dev.txt"
NOENG_FILE = "data_transliterated/rom_dev_noeng.txt"


def load_rom_dev(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = f.readlines()
    d = defaultdict(list)
    for entry in entries:
        rom, dev = entry.strip().split("\t")
        d[rom].append(dev)
    return d


def load_dev_rom(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = f.readlines()
    entries = [e.strip().split("\t") for e in entries]
    return {dev: rom for rom, dev in entries}


def load_dev_rom_single(path):
    d = load_rom_dev(path)
    return {dev[0]: rom for rom, dev in d.items()}


if __name__ == "__main__":
    d = load_dev_rom_single(NOENG_FILE)

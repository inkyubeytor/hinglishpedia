from autocorrect import Speller

spell = Speller(fast=True)


def filter_rom_dev(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        entries = f.readlines()

    with open(out_path, "w+", encoding="utf-8") as f:
        for entry in entries:
            a, b = entry.strip().split("\t")
            if a != spell(a):
                f.write(f"{a}\t{b}\n")


if __name__ == "__main__":
    filter_rom_dev("data_transliterated/rom_dev.txt",
                   "data_transliterated/rom_dev_noeng.txt")

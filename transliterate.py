from util import load_links, title
from google.transliteration import transliterate_word
import time
import random


def get_tokens():
    titles = [title(l) for l in load_links()]
    token_set = set()
    for t in titles:
        with open(f"data_clean/{t}", "r") as f:
            text = f.read()
        tokens = text.split()
        token_set |= {t.strip().lower() for t in tokens if t.strip().isalpha()}

    with open(f"data_transliterated/romanized_tokens.txt", "w+") as f:
        f.write("\n".join(token_set))


def transliterate_tokens():
    n = 0
    with open(f"data_transliterated/romanized_tokens.txt", "r") as f:
        tokens = f.readlines()
    tokens_s = [t.strip() for t in tokens]
    with open(f"data_transliterated/rom_dev.txt", "w+", encoding="utf-8") as f:
        for t in tokens_s:
            try:
                suggestions = transliterate_word(t, lang_code="hi")
            except:
                print(f"Failed to transliterate {t}")
                continue
            for s in suggestions:
                mapping = f"{t}\t{s}"
                n += 1
                print(n, mapping)
                f.write(f"{mapping}\n")
            time.sleep(random.randint(4, 8))


if __name__ == "__main__":
    transliterate_tokens()

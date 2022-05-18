from train_transliterate_model import transliterate, EncoderRNN, AttnDecoderRNN
import torch


MODEL_DIR = "/home2/abhishev/hinglishpedia/models"

ENC = f"{MODEL_DIR}/tl_enc_hs512_e400000.model"
DEC = f"{MODEL_DIR}/tl_dec_hs512_e400000.model"


encoder = torch.load(ENC)
decoder = torch.load(DEC)


def transliterate_sentence(sent):
    return " ".join(w if w.isascii() else transliterate(encoder, decoder, w)
                    for w in sent.split())

def transliterate_file(in_f, out_f, n=None):
    with open(in_f, encoding="utf-8") as f:
        if n is None:
            lines = f.readlines()
        else:
            lines = f.readlines()[:n]
    with open(out_f, "w+", encoding="utf-8") as g:
        for line in lines:
            g.write(transliterate_sentence(line) + "\n")

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 1:
        transliterate_file(sys.argv[1], sys.argv[2])


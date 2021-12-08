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


if __name__ == "__main__":
    import sys
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(in_file, encoding="utf-8") as f:
        lines = f.readlines()

    with open(out_file, encoding="utf-8") as g:
        for line in lines:
            g.write(transliterate_sentence(line) + "\n")

"""
Uses `transliterate` conda env
based on
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
from __future__ import division, print_function, unicode_literals

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from nltk.translate.bleu_score import corpus_bleu

from train_test_split import load_train, load_test, load_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addSeq(self, seq):
        for c in list(seq):
            self.addChar(c)

    def addChar(self, c):
        if c not in self.char2index:
            self.char2index[c] = self.n_chars
            self.char2count[c] = 1
            self.index2char[self.n_chars] = c
            self.n_chars += 1
        else:
            self.char2count[c] += 1


def readLangs(lang1="rom", lang2="dev", reverse=True):
    print("Reading lines...")

    # Read the file and split into lines
    pairs = load_train()

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1="rom", lang2="dev", reverse=True):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s word pairs" % len(pairs))
    print("Counting words...")
    for a, b in load_all():
        if reverse:
            a, b = b, a
        input_lang.addSeq(a)
        output_lang.addSeq(b)
    print("Counted words:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'dev', True)
print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]


def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, epoch_counts, hidden_size, print_every=1000, plot_every=100, learning_rate=0.01):
    n_iters = max(epoch_counts)
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter in epoch_counts:
            torch.save(encoder, f"models/tl_enc_hs{hidden_size}_e{iter}.model")
            torch.save(decoder, f"models/tl_dec_hs{hidden_size}_e{iter}.model")
            word_acc, bleu1, bleu2 = evaluate_test(encoder, decoder)
            with open("models/accs.txt", "a") as f:
                f.write(f"HS: {hidden_size}, Epoch: {iter}, Word Acc: {word_acc}, BLEU-1: {bleu1}, BLEU-2: {bleu2}\n")

    showPlot(plot_losses, n_iters, hidden_size)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points, n_iters, hidden_size):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(f"models/plot_hs{hidden_size}_e{n_iters}.png")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def transliterate(enc, dec, word):
    return "".join(evaluate(enc, dec, word)[0])


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_test(encoder, decoder):
    test_data = load_test()
    correct, total = 0, 0
    refs = []
    cands = []
    for p1, p0 in test_data:
        refs.append([list(p1)])
        total += 1
        out, _ = evaluate(encoder, decoder, p0)
        out_word = "".join(out)
        cands.append(list(out_word))
        if p1 == out_word:
            correct += 1
    word_acc = correct/total
    bleu_1 = corpus_bleu(refs, cands, weights=(1.,))
    bleu_2 = corpus_bleu(refs, cands, weights=(0.5, 0.5))
    print(f"Word Accuracy: {word_acc}, BLEU-1: {bleu_1}, BLEU-2: {bleu_2}")
    return word_acc, bleu_1, bleu_2


def evaluate_show_errors(encoder, decoder):
    test_data = load_test()
    for p1, p0 in test_data:
        out, _ = evaluate(encoder, decoder, p0)
        out_word = "".join(out)
        if p1 != out_word:
            print(f"> {p0}\n= {p1}\n< {out_word}\n")


if __name__ == "__main__":
    epoch_counts = [400_000]
    for hidden_size in [512]:
        encoder1 = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.1).to(device)
    
        trainIters(encoder1, attn_decoder1, epoch_counts, hidden_size, print_every=1000)

    encoder = torch.load("models/tl_enc_hs512_e400000.model")
    decoder = torch.load("models/tl_dec_hs512_e400000.model")
    evaluate_show_errors(encoder, decoder)


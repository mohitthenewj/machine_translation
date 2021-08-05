import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import spacy
import random
import sys
import time
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from inltk.inltk import tokenize
# from inltk.inltk import setup
# setup('hi')
from torchtext import datasets
from torchtext import data


# spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_hi(text):
    return tokenize(text, "hi")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


src = data.Field(tokenize=tokenize_hi)
trg = data.Field(tokenize=tokenize_eng)
mt_train = datasets.TranslationDataset(
     path='./data_torch/data', exts=('.hi', '.en'),
     fields=(src, trg))

mt_test = datasets.TranslationDataset(
     path='./data_torch/data_val', exts=('.hi', '.en'),
     fields=(src, trg))

src.build_vocab(mt_train, max_size=20000, min_freq=2)
trg.build_vocab(mt_train, max_size=20000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(trg.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

num_epochs = 1000
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(src.vocab)
input_size_decoder = len(trg.vocab)
output_size = len(trg.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
# writer = SummaryWriter(f"runs/loss_plot")
step = 0

# print(f'input_size_encoder is {input_size_encoder}')
# print(f'input_size_decoder is {input_size_decoder}')
# print(f'output_size is {output_size}')

# sys.exit(0)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = trg.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


load_model = True
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

model.eval()

# sentence = 'अग्रिम धन राशि इन अस्पतालों को चिकित्सा निरीक्षकों को दी जाएगी जो हर मामले को देखते हुए सहायता प्रदान करेंगे'

st = time.time()

print(f'input_size_encoder is {input_size_encoder}')
print(f'input_size_decoder is {input_size_decoder}')
print(f'output_size is {output_size}')
print(f'pad_idx is {pad_idx}')

print(f'Total time taken for data loading was >> {time.time() -st}')

st = time.time()
score_test = bleu(mt_test, model, src, trg, device)
print(f"Bleu score for test data is {score_test*100:.2f}")

print(f'Total time taken for score_test was >> {time.time() -st}')

st = time.time()
score_train = bleu(mt_train, model, src, trg, device)
print(f"Bleu score for train data is {score_train*100:.2f}")
print(f'Total time taken for score_train was >> {time.time() -st}')

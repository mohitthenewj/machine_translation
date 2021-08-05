import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import re
import pickle
from inltk.inltk import tokenize
from time import time
from tqdm import tqdm

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext import data
from torchtext import datasets

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
from inltk.inltk import tokenize



spacy_eng = spacy.load("en_core_web_sm")
str_punct = '''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।]'''

def tokenize_hi(text):
    return [re.sub(str_punct,'',tok).lower() for tok in tokenize(text, "hi")]


def tokenize_eng(text):
    return [re.sub(str_punct,'',tok.text).lower() for tok in spacy_eng.tokenizer(text)]


hindi = Field(tokenize=tokenize_hi, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)


hindi = data.Field(tokenize=tokenize_hi)
english = data.Field(tokenize=tokenize_eng)
mt_train = datasets.TranslationDataset(
     path='./data_torch/data', exts=('.hi', '.en'),
     fields=(hindi, english))
hindi.build_vocab(mt_train, max_size=15000, min_freq=2)
english.build_vocab(mt_train, max_size=15000, min_freq=2)

# with open('hindi_vocab.pickle', 'wb') as handle:
#     pickle.dump(hindi, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('english_vocab.pickle', 'wb') as handle:
#     pickle.dump(english, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('hindi_vocab.pickle', 'rb') as handle:
#     hindi = pickle.load(handle)

# with open('english_vocab.pickle', 'rb') as handle:
#     english = pickle.load(handle)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training hyperparameters
# num_epochs = 100
learning_rate = 3e-4
# batch_size = 32

# Model hyperparameters
input_size_encoder = len(hindi.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0

print(f'length of input_size_encoder is {input_size_encoder}')

print(f'length of input_size_decoder is {input_size_decoder}')

# Tensorboard to get nice loss plot
# writer = SummaryWriter(f"runs/loss_plot")
step = 0

# train_iterator = data.BucketIterator(
#      dataset=mt_train, batch_size=batch_size,
#      sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device)


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

pad_idx = 1
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# if load_model:
load_model = True
save_model = True

load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
#     spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
#         tokens = [token.text.lower() for token in spacy_ger(sentence)]
        tokens = [i.lower() for i in tokenize(sentence, "hi")]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder, hiddens, cells
            )
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in tqdm(data):
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token
        if len(trg) >0 and len(prediction) >0:
            targets.append([trg])
            outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

mt_test = datasets.TranslationDataset(
     path='./data_torch/data_val', exts=('.hi', '.en'),
     fields=(hindi, english))

score = bleu(mt_test, model, hindi, english, device)

print(f"Bleu score {score * 100:.2f}")


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

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext import data
from torchtext import datasets

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

model_file_name = "checkpoint_trn_v2.pth.tar"
tf_board_path = tf_board_path

spacy_eng = spacy.load("en_core_web_sm")
str_punct = '''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।]'''

def tokenize_hi(text):
    text = re.sub(str_punct,'',text).lower()
    return [tok.lower() for tok in tokenize(text, "hi")]


def tokenize_eng(text):
    text = re.sub(str_punct,'',text).lower()
    return [tok for tok in spacy_eng.tokenizer(text)]

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
#     pickle.dump(hindi, handle)

# with open('english_vocab.pickle', 'wb') as handle:
#     pickle.dump(english, handle)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 1000
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(hindi.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter(tf_board_path)
step = 0

print(f'length of src_vocab_size is {src_vocab_size}')

print(f'length of input_size_decoder is {trg_vocab_size}')


train_iterator = data.BucketIterator(
     dataset=mt_train, batch_size=batch_size,
     sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = 1
# english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "प्रधानमंत्री ने कहा कि भारत में केंद्र सरकार बुनियादी ढांचे पर ध्यान केंद्रित कर रही है।"

################################################################################################################################################

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    str_punct = '''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।]'''
    
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
#         tokens = [token.text.lower() for token in spacy_ger(sentence)]
        sentence = re.sub(str_punct,'',sentence).lower()
        tokens = [re.sub(str_punct,'',tok).lower() for tok in tokenize(sentence, "hi")]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename=model_file_name):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

################################################################################################################################################

for epoch in range(num_epochs):
    st = time()

    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, hindi, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
    print(f'Total time taken for the epoch number {epoch} was {time() - st}')


    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

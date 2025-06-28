import os
import torch
import random
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from model import EncoderRNN, DecoderRNN
from dataloader_generator import *
from utils import train_no_attention, translate

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
MAX_LENGTH = 10
CLIP_GRAD = True
MAX_NORM = 1.0

# Dataset download and preprocessing
if not os.path.exists('fra.txt'):
    os.system('wget -q https://www.manythings.org/anki/fra-eng.zip')
    os.system('unzip -oq fra-eng.zip')

text_pairs = []
for line in open('fra.txt', 'r'):
    a = line.find('CC-BY')
    line = line[:a].strip()
    if '\t' not in line:
        continue
    eng, fra = line.split('\t')
    text_pairs.append((normalizeString(eng), normalizeString(fra)))

input_lang, output_lang, pairs = prepareData('eng', 'fra', text_pairs)
dataset = DatasetEngFra(pairs, input_lang, output_lang)
train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Model setup
encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = DecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)

loss_fn = nn.NLLLoss(ignore_index=PAD_token)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# Training
train_loss = train_no_attention(
    encoder, decoder, train_dl, NUM_EPOCHS, loss_fn,
    encoder_optimizer, decoder_optimizer, device,
    clip_grad=CLIP_GRAD, max_norm=MAX_NORM
)

# Translation Demo
print("\nSample Translations:\n")
for _ in range(10):
    eng, fra = random.choice(text_pairs)
    print("Input:", eng)
    print("Target:", fra)
    print("Predicted:", translate(encoder, decoder, eng, input_lang, output_lang, device), "\n")
    print("#" * 80)

# Plotting
plt.plot(train_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

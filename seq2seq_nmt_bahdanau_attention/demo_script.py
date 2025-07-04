import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader_generator import (
    normalizeString,
    prepareData,
    DatasetEngFra,
    collate_batch,
    PAD_token,
    SOS_token,
    EOS_token
)

from models import EncoderWithBahdanauAttention, DecoderWithBahdanauAttention
from utils import (
    train_bahdanau_attention,
    translate_with_attention,
    plot_attention
)

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 13
torch.manual_seed(seed)

# Hyperparameters
num_epochs = 20
hidden_size = 256
alignment_size = 128
BATCH_SIZE = 32
learning_rate = 1e-3
MAX_LENGTH = 10
alignment_modes = ['concat', 'additive']

# Download dataset if not already present
if not os.path.exists('fra.txt'):
    os.system('wget -q https://www.manythings.org/anki/fra-eng.zip')
    os.system('unzip -oq fra-eng.zip')

# Load and preprocess dataset
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

# Train and evaluate models for each alignment mode
for alignment_mode in alignment_modes:
    print(f"\n{'=' * 40} {alignment_mode.upper()} MODE {'=' * 40}")

    encoder = EncoderWithBahdanauAttention(input_size=input_lang.n_words, hidden_size=hidden_size).to(device)
    decoder = DecoderWithBahdanauAttention(
        output_size=output_lang.n_words,
        hidden_size=hidden_size,
        alignment_size=alignment_size,
        alignment_mode=alignment_mode,
        PAD_token=PAD_token
    ).to(device)

    loss_fn = nn.NLLLoss(ignore_index=PAD_token)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    train_loss = train_bahdanau_attention(
        encoder, decoder, train_dl, num_epochs,
        loss_fn, encoder_optimizer, decoder_optimizer,
        device, output_lang
    )

    print("\nSample Predictions:")
    for _ in range(10):
        eng, fra = random.choice(text_pairs)
        print("Input:", eng)
        print("Target:", fra)
        pred, attentions = translate_with_attention(encoder, decoder, eng, input_lang, output_lang, device)
        print("Predicted:", pred)
        print("#" * 80)

    # Visualize attention for a sample sentence
    sample_input = "Life is often compared to a journey"
    correct_translation = "la vie est souvent comparée à un voyage."
    output_sentence, attention_weights = translate_with_attention(
        encoder, decoder, sample_input, input_lang, output_lang, device
    )
    print("\nPredicted translation:", output_sentence)
    print("Correct translation:", correct_translation)
    plot_attention(sample_input, output_sentence, attention_weights)

    # Plot training loss
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training Loss - {alignment_mode.upper()} Attention")
    plt.grid(True)
    plt.show()

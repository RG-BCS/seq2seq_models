import torch
from torch.utils.data import Dataset
import re
import unicodedata

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10  # max sentence length to filter pairs

class Language_Dictionary_Builder:
    """
    Builds word-to-index and index-to-word mappings with counts.
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD_token', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.n_words = 3  # Count of words, starting after special tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    """
    Convert Unicode string to plain ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters.
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)  # separate punctuation
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)  # remove non-alpha except punctuation
    return s.strip()

def readLangs(lang1, lang2, pairs, reverse=False):
    """
    Prepare input/output languages and optionally reverse pairs.
    """
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language_Dictionary_Builder(lang2)
        output_lang = Language_Dictionary_Builder(lang1)
    else:
        input_lang = Language_Dictionary_Builder(lang1)
        output_lang = Language_Dictionary_Builder(lang2)
    return input_lang, output_lang, pairs

def filterPair(p):
    """
    Filter sentence pairs by max length.
    """
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    """
    Filter all pairs by length.
    """
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, pairs, reverse=False):
    """
    Prepare Language Dictionaries and filter pairs.
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, pairs, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

class DatasetEngFra(Dataset):
    """
    Custom Dataset for English-French sentence pairs.
    Returns tensor pairs of word indices with SOS and EOS tokens for target.
    """
    def __init__(self, eng_fra_data, input_lang, output_lang):
        self.eng_fra = eng_fra_data
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.eng_fra)

    def __getitem__(self, idx):
        eng, fra = self.eng_fra[idx]
        eng_indices = [self.input_lang.word2index[word] for word in eng.split()]
        fra_input = [self.output_lang.word2index[word] for word in fra.split()]
        # Add SOS and EOS tokens to target input sequence
        fra_input.insert(0, SOS_token)
        fra_input.append(EOS_token)
        return torch.tensor(eng_indices, dtype=torch.long), torch.tensor(fra_input, dtype=torch.long)

def collate_batch(batch):
    """
    Collate function for DataLoader.
    Pads input and target sequences, returns lengths for packing.
    """
    eng_list, fra_input, fra_target = [], [], []
    eng_lengths, fra_lengths = [], []

    for eng, fra_i in batch:
        eng_list.append(eng)
        # Target input is fra_i without EOS token (all except last)
        fra_input.append(fra_i[:-1])
        # Target output is fra_i without SOS token (all except first)
        fra_target.append(fra_i[1:])
        eng_lengths.append(len(eng))
        fra_lengths.append(len(fra_i) - 1)  # exclude SOS or EOS appropriately

    eng_pad = torch.nn.utils.rnn.pad_sequence(eng_list, batch_first=True, padding_value=PAD_token)
    fra_input_pad = torch.nn.utils.rnn.pad_sequence(fra_input, batch_first=True, padding_value=PAD_token)
    fra_target_pad = torch.nn.utils.rnn.pad_sequence(fra_target, batch_first=True, padding_value=PAD_token)

    return eng_pad, fra_input_pad, fra_target_pad, torch.tensor(eng_lengths), torch.tensor(fra_lengths)

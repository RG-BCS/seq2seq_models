import torch
from torch.utils.data import Dataset
import re
import unicodedata

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10

class Language_Dictionary_Builder:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD_token', SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3

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
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, pairs, reverse=False):
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language_Dictionary_Builder(lang2)
        output_lang = Language_Dictionary_Builder(lang1)
    else:
        input_lang = Language_Dictionary_Builder(lang1)
        output_lang = Language_Dictionary_Builder(lang2)
    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, pairs, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, pairs, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

class DatasetEngFra(Dataset):
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
        fra_input.insert(0, SOS_token)
        fra_input.append(EOS_token)
        return torch.tensor(eng_indices, dtype=torch.long), torch.tensor(fra_input, dtype=torch.long)

def collate_batch(batch):
    eng_list, fra_input, fra_target = [], [], []
    eng_lengths, fra_lengths = [], []

    for eng, fra_i in batch:
        eng_list.append(eng)
        fra_input.append(fra_i[:-1])
        fra_target.append(fra_i[1:])
        eng_lengths.append(len(eng))
        fra_lengths.append(len(fra_i) - 1)

    eng_pad = torch.nn.utils.rnn.pad_sequence(eng_list, batch_first=True, padding_value=PAD_token)
    fra_input_pad = torch.nn.utils.rnn.pad_sequence(fra_input, batch_first=True, padding_value=PAD_token)
    fra_target_pad = torch.nn.utils.rnn.pad_sequence(fra_target, batch_first=True, padding_value=PAD_token)
    return eng_pad, fra_input_pad, fra_target_pad, torch.tensor(eng_lengths), torch.tensor(fra_lengths)

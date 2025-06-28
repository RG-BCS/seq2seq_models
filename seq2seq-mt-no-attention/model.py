import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_token = 0
SOS_token = 1
EOS_token = 2

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(packed)
        if self.gru.bidirectional:
            hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, hidden, target_input, lengths):
        embedded = self.embedding(target_input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return F.log_softmax(self.out(output), dim=-1), hidden

    def decode_step(self, input_token, hidden):
        embedded = self.embedding(input_token)
        output, hidden = self.gru(embedded, hidden)
        output = F.softmax(self.out(output), dim=-1)
        return output, hidden

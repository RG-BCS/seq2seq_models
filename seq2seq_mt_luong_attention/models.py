import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWithLuongAttention(nn.Module):
    """
    Encoder module for Seq2Seq with Luong-style attention.

    Parameters:
        input_size (int): Size of input vocabulary.
        hidden_size (int): Number of features in hidden state.
        PAD_token (int): Padding index to ignore in embeddings.
    """
    def __init__(self, input_size, hidden_size, PAD_token=0):
        super(EncoderWithLuongAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, eng_inputs, eng_length):
        """
        Args:
            eng_inputs (Tensor): Input tensor of shape [batch_size, seq_len].
            eng_length (Tensor): Actual lengths of each sentence in batch.

        Returns:
            encoder_outputs (Tensor): Padded encoder hidden states [batch_size, seq_len, hidden_size].
            s_0 (Tensor): Final hidden state to be passed to decoder [1, batch_size, hidden_size].
            mask (Tensor): Boolean mask indicating non-PAD positions in input.
        """
        output = self.embedding(eng_inputs)
        packed = nn.utils.rnn.pack_padded_sequence(output, eng_length.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Create mask: [batch_size, max_seq_len]
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len, device=eng_inputs.device)[None, :] < eng_length[:, None]

        # Handle case if GRU is bidirectional
        if self.gru.bidirectional:
            h_n_forward = hidden[-2]
            h_n_backward = hidden[-1]
            s_0 = torch.cat((h_n_forward, h_n_backward), dim=1)
        else:
            s_0 = hidden

        return encoder_outputs, s_0, mask


class DecoderWithLuongAttention(nn.Module):
    """
    Decoder with Luong attention mechanism.

    Parameters:
        output_size (int): Size of output vocabulary.
        hidden_size (int): Decoder hidden size.
        alignment_mode (str): Type of attention ['dot_product', 'general', 'concat'].
        dropout (float): Dropout rate.
    """
    def __init__(self, output_size, hidden_size, alignment_mode='dot_product', dropout=0.1):
        super(DecoderWithLuongAttention, self).__init__()
        assert alignment_mode in ['dot_product', 'general', 'concat'], \
            f"Invalid alignment_mode: {alignment_mode}. Must be one of ['dot_product', 'general', 'concat']"

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alignment_mode = alignment_mode

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn_cell = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Alignment-specific layers
        if self.alignment_mode == 'general':
            self.w = nn.Linear(hidden_size, hidden_size)
        elif self.alignment_mode == 'concat':
            self.w = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

        self.concat_hidden_context = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, all_encoder_hidden_states, initial_decoder_hidden_state,
                encoder_output_mask, target_input, fra_length):
        """
        Forward pass for training.

        Args:
            all_encoder_hidden_states (Tensor): Encoder outputs [batch, seq_len, hidden].
            initial_decoder_hidden_state (Tensor): Initial decoder hidden state [1, batch, hidden].
            encoder_output_mask (Tensor): Mask for PAD tokens in encoder output [batch, seq_len].
            target_input (Tensor): Decoder input tokens [batch, target_len].
            fra_length (Tensor): Actual lengths of target sequences.

        Returns:
            log_probs (Tensor): Log probabilities [batch, target_len, output_vocab_size].
        """
        decoder_hidden = initial_decoder_hidden_state
        embedded_input = self.embedding(target_input)
        batch, seq_len, _ = embedded_input.shape
        logits = []

        for t in range(seq_len):
            active_mask = (t < fra_length).float().unsqueeze(1)
            if active_mask.sum() == 0:
                continue

            decoder_hidden_old = decoder_hidden
            input_t = embedded_input[:, t, :]
            rnn_out, decoder_hidden = self.rnn_cell(input_t.unsqueeze(1), decoder_hidden)

            context_vector, attention_weights = self.compute_context_vectors(
                decoder_hidden, all_encoder_hidden_states, encoder_output_mask
            )
            hidden_for_output = torch.tanh(
                self.concat_hidden_context(torch.cat((context_vector, decoder_hidden), dim=-1))
            )
            logit_t = self.output_layer(hidden_for_output)
            logits.append(logit_t.squeeze(0))

            decoder_hidden = decoder_hidden * active_mask.unsqueeze(0) + decoder_hidden_old * (1 - active_mask.unsqueeze(0))

        logits = torch.stack(logits, dim=1)
        return F.log_softmax(logits, dim=-1)

    def compute_context_vectors(self, decoder_hidden, encoder_outputs, encoder_output_mask=None):
        """
        Computes context vector using selected attention mechanism.
        """
        if self.alignment_mode == 'dot_product':
            hidden = decoder_hidden.squeeze(0).unsqueeze(2)
            alignment_scores = torch.bmm(encoder_outputs, hidden).squeeze(2)
            if encoder_output_mask is not None:
                alignment_scores = alignment_scores.masked_fill(~encoder_output_mask, float('-inf'))
            alphas = torch.softmax(alignment_scores, dim=-1).unsqueeze(-1)
            context_vector = torch.sum(alphas * encoder_outputs, dim=1)
            return context_vector.unsqueeze(0), alphas.squeeze(-1)

        elif self.alignment_mode == 'general':
            hidden_reshaped = decoder_hidden.squeeze(0)
            hidden_transformed = self.w(hidden_reshaped).unsqueeze(2)
            alignment_scores = torch.bmm(encoder_outputs, hidden_transformed).squeeze(2)
            if encoder_output_mask is not None:
                alignment_scores = alignment_scores.masked_fill(~encoder_output_mask, float('-inf'))
            alphas = torch.softmax(alignment_scores, dim=-1).unsqueeze(-1)
            context_vector = torch.sum(alphas * encoder_outputs, dim=1)
            return context_vector.unsqueeze(0), alphas.squeeze(-1)

        elif self.alignment_mode == 'concat':
            batch, seq_len, _ = encoder_outputs.size()
            hidden_expanded = decoder_hidden.squeeze(0).unsqueeze(1).expand(-1, seq_len, -1)
            concat_input = torch.cat([hidden_expanded, encoder_outputs], dim=-1)
            alignment_scores = self.v(torch.tanh(self.w(concat_input))).squeeze(2)
            if encoder_output_mask is not None:
                alignment_scores = alignment_scores.masked_fill(~encoder_output_mask, float('-inf'))
            alphas = torch.softmax(alignment_scores, dim=-1).unsqueeze(-1)
            context_vector = torch.sum(alphas * encoder_outputs, dim=1)
            return context_vector.unsqueeze(0), alphas.squeeze(-1)

    def decode_step(self, input_token, decoder_hidden, encoder_outputs, encoder_output_mask):
        """
        Performs a single decoding step (used during inference).

        Args:
            input_token (Tensor): Last predicted token [batch, 1].
            decoder_hidden (Tensor): Current decoder hidden state.
            encoder_outputs (Tensor): Encoder output features.
            encoder_output_mask (Tensor): Padding mask for encoder outputs.

        Returns:
            probs (Tensor): Softmax probabilities over vocab.
            decoder_hidden_new (Tensor): Updated hidden state.
            attention_weights (Tensor): Attention weights over input sequence.
        """
        embedded = self.embedding(input_token)
        _, decoder_hidden_new = self.rnn_cell(embedded, decoder_hidden)
        context_vector, attention_weights = self.compute_context_vectors(
            decoder_hidden_new, encoder_outputs, encoder_output_mask
        )
        hidden_for_output = torch.tanh(self.concat_hidden_context(torch.cat((context_vector, decoder_hidden_new), dim=-1)))
        logits = self.output_layer(hidden_for_output)
        probs = F.softmax(logits, dim=-1)
        return probs, decoder_hidden_new, attention_weights

import torch
import torch.nn as nn

class EncoderWithBahdanauAttention(nn.Module):
    """
    Encoder module for Seq2Seq model with Bahdanau Attention.

    This encoder uses a GRU over embedded input sequences.
    It outputs:
    - encoder_outputs: All hidden states for each input token (padded sequences).
    - hidden: Final hidden state for the encoder (used to initialize decoder).
    - mask: Boolean mask to indicate valid (non-padding) tokens per batch.
    """

    def __init__(self, input_size, hidden_size, PAD_token=0):
        """
        Args:
            input_size (int): Size of the input vocabulary.
            hidden_size (int): Number of features in the hidden state of the GRU.
            PAD_token (int): Index used for padding tokens in the input sequences.
        """
        super(EncoderWithBahdanauAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Embedding layer: converts token indices into embeddings
        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=hidden_size,
                                      padding_idx=PAD_token)
        # GRU layer: processes embedded inputs
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, eng_inputs, eng_length):
        """
        Forward pass for the encoder.

        Args:
            eng_inputs (Tensor): Batch of input sequences [batch_size, seq_len].
            eng_length (Tensor): Lengths of sequences before padding [batch_size].

        Returns:
            encoder_outputs (Tensor): Outputs for each time step [batch, seq_len, hidden_size].
            hidden (Tensor): Final hidden state [1, batch, hidden_size].
            mask (Tensor): Boolean mask for valid tokens [batch, seq_len].
        """
        # Embed the input tokens
        embedded = self.embedding(eng_inputs)  # shape: [batch_size, seq_len, hidden_size]

        # Pack padded sequence for efficient processing of variable lengths
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, eng_length.cpu().numpy(), batch_first=True, enforce_sorted=False)

        # Pass through GRU
        packed_outputs, hidden = self.gru(packed_embedded)

        # Unpack the sequence back to padded form
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Create mask for valid tokens (non-padding) based on sequence lengths
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len, device=eng_inputs.device)[None, :] < eng_length[:, None]

        return encoder_outputs, hidden, mask


class DecoderWithBahdanauAttention(nn.Module):
    """
    Decoder module with Bahdanau-style attention.

    Supports two alignment modes:
    - 'concat': concatenation-based alignment
    - 'additive': additive alignment (original Bahdanau)

    The decoder predicts target tokens using teacher forcing during training.
    """

    def __init__(self, output_size, hidden_size, alignment_size,
                 alignment_mode='concat', PAD_token=0):
        """
        Args:
            output_size (int): Size of target vocabulary.
            hidden_size (int): Number of features in hidden states.
            alignment_size (int): Size of the intermediate alignment layer.
            alignment_mode (str): 'concat' or 'additive'.
            PAD_token (int): Padding token index for target sequences.
        """
        super(DecoderWithBahdanauAttention, self).__init__()
        assert alignment_mode in ['concat', 'additive'], f"Unknown alignment mode: {alignment_mode}"

        self.alignment_mode = alignment_mode

        # Embedding layer for target tokens
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_token)

        # GRU input is concatenation of embedded previous token and context vector
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)

        # Output layers: combine decoder hidden state and context vector
        self.out_w = nn.Linear(hidden_size * 2, hidden_size)
        self.out_v = nn.Linear(hidden_size, output_size)

        # Layers for alignment scoring
        if alignment_mode == 'concat':
            self.w = nn.Linear(hidden_size * 2, alignment_size)
            self.v = nn.Linear(alignment_size, 1)
        else:  # additive
            self.w_encoder = nn.Linear(hidden_size, alignment_size)
            self.w_decoder = nn.Linear(hidden_size, alignment_size)
            self.v = nn.Linear(alignment_size, 1)

    def forward(self, all_encoder_hidden_states, initial_decoder_hidden_state,
                encoder_output_mask, target_input, fra_length):
        """
        Forward pass through the decoder with teacher forcing.

        Args:
            all_encoder_hidden_states (Tensor): Encoder outputs [batch, seq_len, hidden_size].
            initial_decoder_hidden_state (Tensor): Initial hidden state [1, batch, hidden_size].
            encoder_output_mask (Tensor): Mask for encoder outputs [batch, seq_len].
            target_input (Tensor): Target sequences input [batch, target_seq_len].
            fra_length (Tensor): Lengths of target sequences [batch].

        Returns:
            log_probs (Tensor): Log probabilities of predictions [batch, seq_len, output_size].
        """
        decoder_hidden = initial_decoder_hidden_state  # [1, batch, hidden_size]
        embedded_input = self.embedding(target_input)  # [batch, seq_len, hidden_size]

        batch_size, seq_len_fre, _ = embedded_input.shape
        logits = []

        for t in range(seq_len_fre):
            # Determine which sequences are still active (not padding at this timestep)
            active_mask = (t < fra_length).float().unsqueeze(1)  # [batch, 1]
            if active_mask.sum() == 0:
                # All sequences finished, no need to continue decoding
                continue

            # Compute context vector using attention
            context_vector, _ = self.compute_context_vectors(
                decoder_hidden, all_encoder_hidden_states, encoder_output_mask)

            # Get embedding for current input token
            y_t = embedded_input[:, t, :]  # [batch, hidden_size]

            # Concatenate input embedding and context vector for GRU input
            gru_input = torch.cat([y_t, context_vector], dim=1).unsqueeze(1)  # [batch, 1, 2*hidden_size]

            # Pass through GRU
            _, next_state = self.gru(gru_input, decoder_hidden)

            # Update hidden state only for active sequences
            decoder_hidden = (active_mask.unsqueeze(0) * next_state +
                              (1 - active_mask).unsqueeze(0) * decoder_hidden)

            # Concatenate decoder hidden and context for output layer
            logit_t = self.out_w(torch.cat([decoder_hidden.squeeze(0), context_vector], dim=-1))
            logit_t = self.out_v(torch.tanh(logit_t))

            logits.append(logit_t)  # [batch, output_size]

        # Stack logits for all timesteps [batch, seq_len, output_size]
        logits = torch.stack(logits, dim=1)

        # Return log softmax for stable training with NLLLoss
        return nn.functional.log_softmax(logits, dim=-1)

    def compute_context_vectors(self, decoder_hidden, all_encoder_hidden_states, encoder_output_mask=None):
        """
        Compute context vector and attention weights for current decoder hidden state.

        Args:
            decoder_hidden (Tensor): Current hidden state of decoder [1, batch, hidden_size].
            all_encoder_hidden_states (Tensor): Encoder outputs [batch, seq_len, hidden_size].
            encoder_output_mask (Tensor, optional): Mask for padding positions [batch, seq_len].

        Returns:
            context (Tensor): Context vector computed as weighted sum [batch, hidden_size].
            attention_weights (Tensor): Attention weights [batch, seq_len].
        """
        if self.alignment_mode == 'concat':
            batch, seq_len, _ = all_encoder_hidden_states.size()

            # Expand decoder hidden state to match encoder seq length
            dec_h_expanded = decoder_hidden.squeeze(0).unsqueeze(1).expand(-1, seq_len, -1)

            # Concatenate encoder states and decoder hidden states for scoring
            concat = torch.cat((all_encoder_hidden_states, dec_h_expanded), dim=-1)  # [batch, seq_len, 2*hidden_size]

            energy = torch.tanh(self.w(concat))  # [batch, seq_len, alignment_size]
            logits = self.v(energy).squeeze(-1)  # [batch, seq_len]

            # Mask padding tokens
            if encoder_output_mask is not None:
                assert encoder_output_mask.shape == logits.shape, "Mask shape mismatch!"
                logits = logits.masked_fill(~encoder_output_mask, float('-inf'))

            alphas = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
            context = torch.sum(alphas * all_encoder_hidden_states, dim=1)  # [batch, hidden_size]
            return context, alphas.squeeze(-1)

        else:  # additive
            batch, seq_len, _ = all_encoder_hidden_states.size()

            encoder_proj = self.w_encoder(all_encoder_hidden_states)  # [batch, seq_len, alignment_size]
            decoder_proj = self.w_decoder(decoder_hidden)             # [1, batch, alignment_size]

            # Expand decoder projection to seq length dimension
            decoder_proj_expanded = decoder_proj.squeeze(0).unsqueeze(1).expand(-1, seq_len, -1)

            sum_proj = torch.tanh(encoder_proj + decoder_proj_expanded)  # [batch, seq_len, alignment_size]
            logits = self.v(sum_proj).squeeze(-1)                         # [batch, seq_len]

            # Mask padding tokens
            if encoder_output_mask is not None:
                assert encoder_output_mask.shape == logits.shape, "Mask shape mismatch!"
                logits = logits.masked_fill(~encoder_output_mask, float('-inf'))

            alphas = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
            context = torch.sum(alphas * all_encoder_hidden_states, dim=1)  # [batch, hidden_size]
            return context, alphas.squeeze(-1)

    def decode_step(self, input_token, decoder_hidden, encoder_outputs, encoder_output_mask):
        """
        Perform a single decoding step for inference.

        Args:
            input_token (Tensor): Input token indices [batch, 1].
            decoder_hidden (Tensor): Previous hidden state [1, batch, hidden_size].
            encoder_outputs (Tensor): Encoder hidden states [batch, seq_len, hidden_size].
            encoder_output_mask (Tensor): Mask for encoder outputs [batch, seq_len].

        Returns:
            probs (Tensor): Probability distribution over output vocabulary [batch, output_size].
            new_hidden (Tensor): Updated decoder hidden state [1, batch, hidden_size].
            attention_weights (Tensor): Attention weights [batch, seq_len].
        """
        # Embed the current input token
        embedded = self.embedding(input_token).squeeze(1)  # [batch, hidden_size]

        # Compute context vector and attention weights
        context_vector, attention_weights = self.compute_context_vectors(
            decoder_hidden, encoder_outputs, encoder_output_mask)

        # Concatenate embedding and context as input to GRU
        rnn_input = torch.cat([embedded, context_vector], dim=-1).unsqueeze(1)

        # Pass through GRU
        output, new_hidden = self.gru(rnn_input, decoder_hidden)

        # Generate logits from new hidden state and context
        logit_t = self.out_w(torch.cat([new_hidden.squeeze(0), context_vector], dim=-1))
        logit_t = self.out_v(torch.tanh(logit_t))

        # Convert logits to probabilities
        probs = nn.functional.softmax(logit_t, dim=-1)

        return probs, new_hidden, attention_weights

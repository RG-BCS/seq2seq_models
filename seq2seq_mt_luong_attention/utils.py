import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


PAD_token, SOS_token, EOS_token = 0, 1, 2


def grad_norm(model):
    """
    Computes total gradient norm for a given model.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def translate_with_attention(encoder, decoder, sentence, input_lang, output_lang, max_len=10):
    """
    Translates an input sentence using the trained encoder and decoder with Luong attention.

    Args:
        encoder (nn.Module): Trained encoder model.
        decoder (nn.Module): Trained decoder model.
        sentence (str): English input sentence.
        input_lang: Source language dictionary.
        output_lang: Target language dictionary.
        max_len (int): Max output length.

    Returns:
        Tuple of (translated_sentence, attention_weights).
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        eng_inds = [input_lang.word2index.get(word, 0) for word in sentence.strip().split()]
        eng_tensor = torch.tensor(eng_inds, dtype=torch.long).unsqueeze(0).to(encoder.embedding.weight.device)
        input_length_tensor = torch.tensor([len(eng_inds)], dtype=torch.long).to(eng_tensor.device)

        encoder_outputs, encoder_final_hidden, mask = encoder(eng_tensor, input_length_tensor)
        decoder_hidden = encoder_final_hidden
        next_token = torch.tensor([[SOS_token]], dtype=torch.long, device=eng_tensor.device)

        output_tokens = []
        attentions = []

        for _ in range(max_len):
            probs, decoder_hidden, attention_weights = decoder.decode_step(next_token, decoder_hidden, encoder_outputs, mask)
            next_token = torch.argmax(probs.squeeze(0), dim=-1).unsqueeze(0)
            attentions.append(attention_weights.cpu().numpy())
            token_id = next_token.item()
            if token_id == EOS_token:
                break
            output_tokens.append(output_lang.index2word.get(token_id, '<UNK>'))

        attentions = np.stack(attentions).squeeze(1)
        return " ".join(output_tokens), attentions


def plot_attention(input_sentence, output_sentence, attention):
    """
    Plots the attention matrix between input and output tokens.

    Args:
        input_sentence (str): Input sentence.
        output_sentence (str): Translated output sentence.
        attention (np.ndarray): Attention weights.
    """
    input_tokens = input_sentence.strip().split()
    output_tokens = output_sentence.strip().split()

    fig, ax = plt.subplots(figsize=(min(12, len(input_tokens) * 0.8), min(12, len(output_tokens) * 0.8)))
    cax = ax.matshow(attention, cmap='viridis')

    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens, rotation=90, fontsize=10)

    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens, fontsize=10)

    fig.colorbar(cax)
    ax.set_xlabel("Input Sentence", fontsize=12)
    ax.set_ylabel("Output Sentence", fontsize=12)
    plt.tight_layout()
    plt.show()


def train_luong_attention(encoder, decoder, train_dl, num_epochs, loss_fn,
                          encoder_optimizer, decoder_optimizer, clip_norm=False, max_norm=1.0):
    """
    Training loop for Luong attention-based seq2seq model.

    Args:
        encoder (nn.Module)
        decoder (nn.Module)
        train_dl (DataLoader): Training data loader
        num_epochs (int)
        loss_fn: Loss function
        encoder_optimizer, decoder_optimizer: Optimizers
        clip_norm (bool): Enable gradient clipping
        max_norm (float): Max norm for clipping

    Returns:
        List of training losses
    """
    encoder.train()
    decoder.train()
    start = time.time()
    total_loss = []

    for epoch in range(num_epochs):
        batch_loss = 0.0
        for eng, fre_input, fre_target, eng_leng, fre_length in train_dl:
            eng, fre_input, fre_target = eng.to(encoder.embedding.weight.device), fre_input.to(encoder.embedding.weight.device), fre_target.to(encoder.embedding.weight.device)
            eng_leng, fre_length = eng_leng.to(eng.device), fre_length.to(eng.device)

            all_encoder_hidden_states, initial_decoder_hidden_state, mask = encoder(eng, eng_leng)
            pred = decoder(all_encoder_hidden_states, initial_decoder_hidden_state, mask, fre_input, fre_length)

            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), fre_target.reshape(-1))
            loss.backward()

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=max_norm)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=max_norm)

            norm_grad_encoder = grad_norm(encoder)
            norm_grad_decoder = grad_norm(decoder)

            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            batch_loss += loss.item() * eng.size(0)

        batch_loss /= len(train_dl.dataset)
        total_loss.append(batch_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print_epoch_info(epoch, num_epochs, batch_loss, norm_grad_encoder, norm_grad_decoder, start, encoder, decoder)
            start = time.time()

    return total_loss


def print_epoch_info(epoch, num_epochs, loss, norm_grad_encoder, norm_grad_decoder, start_time,
                     encoder, decoder, sample_input="Life is often compared to a journey",
                     correct_translation="la vie est souvent comparée à un voyage."):
    """
    Prints training progress and shows a sample translation.
    """
    elapsed_min = (time.time() - start_time) / 60
    print(f'Epoch {epoch}/{num_epochs} | Loss: {loss:.4f} | Encoder_Norm_Grad: {norm_grad_encoder:.3f} | '
          f'Decoder_Norm_Grad: {norm_grad_decoder:.3f} | Time: {elapsed_min:.4f} min')

    output_sentence, attention_weights = translate_with_attention(encoder, decoder, sample_input, input_lang, output_lang)
    print("\nPredicted translation:", output_sentence)
    print("Correct translation:", correct_translation, end='\n\n')
    encoder.train()
    decoder.train()

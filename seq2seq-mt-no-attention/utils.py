import torch
import time

PAD_token = 0
SOS_token = 1
EOS_token = 2

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_no_attention(
    encoder,
    decoder,
    train_dl,
    num_epochs,
    loss_fn,
    encoder_optimizer,
    decoder_optimizer,
    device,
    clip_grad=False,
    max_norm=1.0
):
    encoder.train()
    decoder.train()
    total_loss = []

    for epoch in range(num_epochs):
        batch_loss = 0.0
        start = time.time()

        for eng, fra_input, fra_target, eng_lens, fra_lens in train_dl:
            eng = eng.to(device)
            fra_input = fra_input.to(device)
            fra_target = fra_target.to(device)
            eng_lens = eng_lens.to(device)
            fra_lens = fra_lens.to(device)

            _, hidden = encoder(eng, eng_lens)
            output, _ = decoder(hidden, fra_input, fra_lens)

            loss = loss_fn(output.reshape(-1, output.shape[-1]), fra_target.reshape(-1))
            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            batch_loss += loss.item() * eng.size(0)

        avg_loss = batch_loss / len(train_dl.dataset)
        total_loss.append(avg_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            grad_norm_encoder = grad_norm(encoder)
            grad_norm_decoder = grad_norm(decoder)
            print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Encoder Grad Norm: {grad_norm_encoder:.3f} | "
                  f"Decoder Grad Norm: {grad_norm_decoder:.3f} | "
                  f"Time: {time.time() - start:.2f}s")

    return total_loss

def translate(encoder, decoder, sentence, input_lang, output_lang, device, max_len=10):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        indices = [input_lang.word2index.get(word, 0) for word in sentence.split()]
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = encoder(input_tensor, torch.tensor([len(indices)]).to(device))

        output_words = []
        next_token = torch.tensor([[SOS_token]], device=device)

        for _ in range(max_len):
            pred, hidden = decoder.decode_step(next_token, hidden)
            next_token = torch.argmax(pred, dim=-1)
            if next_token.item() == EOS_token:
                break
            output_words.append(output_lang.index2word.get(next_token.item(), "<UNK>"))

        return " ".join(output_words)

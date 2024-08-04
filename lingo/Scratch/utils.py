import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = torch.tensor(encoded).unsqueeze(0)

    return encoded


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def generate_text(model, input_ids, max_tokens, context_length):
    for _ in range(max_tokens):
        context = input_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :]

        next_token_index = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token_index), dim=1)

    return input_ids

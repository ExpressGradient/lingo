import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = torch.tensor(encoded).unsqueeze(0)

    return encoded


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def generate_text(
    model, input_ids, max_tokens, context_length, temperature=0.0, top_k=None
):
    for _ in range(max_tokens):
        context = input_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token_index = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token_index = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token_index), dim=1)

    return input_ids

import torch
import torch.nn.functional as F


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


def compute_batch_loss(input_batch, target_batch, model):
    input_batch, target_batch = input_batch.to("cuda"), target_batch.to("cuda")
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss


def compute_loader_loss(loader, model, num_batches):
    total_loss = 0.0
    num_batches = min(num_batches, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = compute_batch_loss(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(train_loader, valid_loader, model, eval_iter):
    model.eval()

    with torch.no_grad():
        train_loss = compute_loader_loss(train_loader, model, eval_iter)
        valid_loss = compute_loader_loss(valid_loader, model, eval_iter)

    model.train()

    return train_loss, valid_loss

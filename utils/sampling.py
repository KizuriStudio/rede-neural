import numpy as np


def top_p_sample(logits, p=0.9, temperature=1.0):
    """Nucleus sampling (top-p) — mantém só os tokens que somam p% da probabilidade.
    Usado em ChatGPT, Claude, LLaMA."""
    logits = logits / max(temperature, 1e-8)
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()

    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    mask = cumulative <= p
    mask[0] = True
    top_idx = sorted_idx[mask]

    top_probs = probs[top_idx]
    top_probs = top_probs / top_probs.sum()

    return np.random.choice(top_idx, p=top_probs)


def top_k_sample(logits, k=40, temperature=1.0):
    """Top-k sampling — mantém os k tokens mais prováveis."""
    logits = logits / max(temperature, 1e-8)
    top_k = min(k, len(logits))
    top_idx = np.argsort(-logits)[:top_k]
    top_logits = logits[top_idx]
    top_logits = top_logits - top_logits.max()
    probs = np.exp(top_logits) / np.exp(top_logits).sum()
    return np.random.choice(top_idx, p=probs)


def repetition_penalty_fn(logits, generated_tokens, penalty=1.2):
    """Repetition penalty — penaliza tokens que já aparecerem.
    Usado em todos os LLMs modernos pra evitar repetição."""
    logits = logits.copy()
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def generate_advanced(model, dataset, seed_text, length=500, temperature=1.0,
                       top_k=0, top_p=1.0, repetition_penalty=1.0):
    """Geração avançada com top-p, top-k e repetition penalty."""
    model.eval()
    model.clear_cache()

    input_ids = dataset.encode(seed_text)
    generated = list(input_ids)

    for _ in range(length):
        x = np.array([generated[-128:] if len(generated) >= 128 else generated], dtype=np.int32)
        if x.shape[1] < 128:
            padding = np.zeros((1, 128 - x.shape[1]), dtype=np.int32)
            x = np.concatenate([padding, x], axis=1)

        logits = model.forward(x)
        next_logits = logits[0, -1, :]
        next_logits = repetition_penalty_fn(next_logits, generated, repetition_penalty)

        if top_p < 1.0:
            token_id = top_p_sample(next_logits, p=top_p, temperature=temperature)
        elif top_k > 0:
            token_id = top_k_sample(next_logits, k=top_k, temperature=temperature)
        else:
            token_id = np.random.choice(len(next_logits))

        generated.append(int(token_id))

    model.train()
    model.clear_cache()
    return dataset.decode(generated)


def generate_stream(model, dataset, seed_text, length=500, temperature=1.0,
                     top_k=0, top_p=1.0, repetition_penalty=1.0):
    """Gera texto char por char no terminal."""
    import sys

    model.eval()
    model.clear_cache()

    input_ids = dataset.encode(seed_text)
    generated = list(input_ids)

    sys.stdout.write(seed_text)
    sys.stdout.flush()

    for _ in range(length):
        x = np.array([generated[-128:] if len(generated) >= 128 else generated], dtype=np.int32)
        if x.shape[1] < 128:
            padding = np.zeros((1, 128 - x.shape[1]), dtype=np.int32)
            x = np.concatenate([padding, x], axis=1)

        logits = model.forward(x)
        next_logits = logits[0, -1, :]
        next_logits = repetition_penalty_fn(next_logits, generated, repetition_penalty)

        if top_p < 1.0:
            token_id = top_p_sample(next_logits, p=top_p, temperature=temperature)
        elif top_k > 0:
            token_id = top_k_sample(next_logits, k=top_k, temperature=temperature)
        else:
            token_id = np.random.choice(len(next_logits))

        generated.append(int(token_id))
        char = dataset.idx_to_char.get(int(token_id), "?")
        sys.stdout.write(char)
        sys.stdout.flush()

    model.train()
    model.clear_cache()
    print()

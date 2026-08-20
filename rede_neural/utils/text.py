import numpy as np


def generate_text(model, dataset, seed_text, length=200, temperature=0.8, top_k=0):
    """Gera texto a partir de um seed."""

    model.eval()
    input_seq = dataset.encode(seed_text)

    generated = list(input_seq)

    for _ in range(length):
        x = np.array([generated[-64:] if len(generated) >= 64 else generated], dtype=np.int32)

        if x.shape[1] < 64:
            padding = np.zeros((1, 64 - x.shape[1]), dtype=np.int32)
            x = np.concatenate([padding, x], axis=1)

        logits = model.forward(x)
        next_logits = logits[0, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            top_indices = np.argsort(next_logits)[-top_k:]
            top_logits = next_logits[top_indices]
            exp_logits = np.exp(top_logits - top_logits.max())
            probs = exp_logits / exp_logits.sum()
            chosen = np.random.choice(top_indices, p=probs)
        else:
            exp_logits = np.exp(next_logits - next_logits.max())
            probs = exp_logits / exp_logits.sum()
            chosen = np.random.choice(len(probs), p=probs)

        generated.append(chosen)

    model.train()
    return dataset.decode(generated)


def generate_interactive(model, dataset, seed="O", length=500, temperature=0.7, top_k=40):
    """Gera texto com output no terminal em tempo real."""
    import sys

    model.eval()
    input_seq = dataset.encode(seed)
    generated = list(input_seq)

    sys.stdout.write(seed)
    sys.stdout.flush()

    for _ in range(length):
        x = np.array([generated[-64:] if len(generated) >= 64 else generated], dtype=np.int32)
        if x.shape[1] < 64:
            padding = np.zeros((1, 64 - x.shape[1]), dtype=np.int32)
            x = np.concatenate([padding, x], axis=1)

        logits = model.forward(x)
        next_logits = logits[0, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            top_indices = np.argsort(next_logits)[-top_k:]
            top_logits = next_logits[top_indices]
            exp_logits = np.exp(top_logits - top_logits.max())
            probs = exp_logits / exp_logits.sum()
            chosen = np.random.choice(top_indices, p=probs)
        else:
            exp_logits = np.exp(next_logits - next_logits.max())
            probs = exp_logits / exp_logits.sum()
            chosen = np.random.choice(len(probs), p=probs)

        generated.append(chosen)
        char = dataset.idx_to_char.get(chosen, "?")
        sys.stdout.write(char)
        sys.stdout.flush()

    model.train()
    print()

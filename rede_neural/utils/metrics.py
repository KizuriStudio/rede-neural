import numpy as np


def accuracy(predictions, targets):
    """Accuracy classification."""
    if predictions.ndim > 1:
        pred = np.argmax(predictions, axis=-1)
    else:
        pred = (predictions > 0.5).astype(int)
    return (pred == targets).mean()


def top_k_accuracy(predictions, targets, k=5):
    """Top-K accuracy."""
    batch = predictions.shape[0]
    top_k = np.argsort(predictions, axis=-1)[:, -k:]
    hits = 0
    for i in range(batch):
        if targets[i] in top_k[i]:
            hits += 1
    return hits / batch


def perplexity(loss):
    """Perplexity = exp(loss) — métrica padrão pra language models."""
    return np.exp(loss)


def bleu_score(predicted, reference, max_n=4):
    """BLEU score simplificado pra avaliação de texto."""
    from collections import Counter

    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter([tuple(predicted[i:i+n]) for i in range(len(predicted) - n + 1)])
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)])

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = max(sum(pred_ngrams.values()), 1)
        scores.append(overlap / total if total > 0 else 0)

    if min(scores) > 0:
        geometric_mean = np.exp(np.mean(np.log(scores)))
    else:
        geometric_mean = 0

    brevity = min(1.0, len(predicted) / max(len(reference), 1))
    return geometric_mean * brevity


def cosine_similarity(a, b):
    """Similaridade cosseno entre dois vetores."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-8)

import numpy as np


def precompute_rope_freqs(dim, max_len=512, theta=10000.0):
    """Precomputa frequencias do RoPE (Rotary Position Embedding).
    Usado em LLaMA, GPT-NeoX, Qwen."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(float) / dim))
    t = np.arange(max_len)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_rope(x, cos, sin):
    """Aplica RoPE nos tensores Q e K.
    x: (batch, seq_len, num_heads, head_dim)"""
    B, T, H, D = x.shape
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]

    cos_t = cos[:T, :D // 2].reshape(1, T, 1, D // 2)
    sin_t = sin[:T, :D // 2].reshape(1, T, 1, D // 2)

    out1 = x1 * cos_t - x2 * sin_t
    out2 = x2 * cos_t + x1 * sin_t

    return np.concatenate([out1, out2], axis=-1)


class RoPE:
    """Wrapper que precomputa e aplica Rotary Position Embedding."""

    def __init__(self, head_dim, max_len=512, theta=10000.0):
        self.head_dim = head_dim
        self.cos, self.sin = precompute_rope_freqs(head_dim, max_len, theta)

    def apply(self, x):
        return apply_rope(x, self.cos, self.sin)

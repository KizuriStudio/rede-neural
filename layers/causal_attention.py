import numpy as np
from layers.dense import Layer
from .rope import RoPE


class CausalSelfAttention(Layer):
    """Multi-Head Self-Attention com causal mask e KV cache.
    Usado em GPT-2, GPT-3, LLaMA, Mistral, Claude."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512, use_rope=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_len = max_len
        self.use_rope = use_rope

        scale = np.sqrt(2.0 / embed_dim)
        self.params["Wq"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wk"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wv"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wo"] = np.random.randn(embed_dim, embed_dim) * scale

        if use_rope:
            self.rope = RoPE(self.head_dim, max_len)

        causal = np.triu(np.ones((max_len, max_len), dtype=bool), k=1)
        self._causal_mask = causal

        self.dropout_p = dropout
        self._kv_cache = None

    def forward(self, x, use_cache=False, pos_offset=0):
        B, T, D = x.shape

        Q = (x @ self.params["Wq"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.params["Wk"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.params["Wv"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_rope:
            Q = self.rope.apply(Q)
            K = self.rope.apply(K)

        if use_cache and self._kv_cache is not None:
            K_cache, V_cache = self._kv_cache
            K = np.concatenate([K_cache, K], axis=2)
            V = np.concatenate([V_cache, V], axis=2)

        if use_cache:
            self._kv_cache = (K.copy(), V.copy())

        T_full = K.shape[2]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)

        causal = self._causal_mask[pos_offset:pos_offset + T, :T_full]
        scores = scores + causal[np.newaxis, np.newaxis, :, :] * (-1e9)

        if self._training and self.dropout_p > 0:
            mask = (np.random.rand(*scores.shape) > self.dropout_p) / (1 - self.dropout_p)
            scores = scores * mask

        attn = self._softmax(scores)
        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, D)
        out = out @ self.params["Wo"]

        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "attn": attn, "B": B, "T": T}
        return out

    def backward(self, grad):
        B, T, D = grad.shape
        cache = self._cache

        x_flat = cache["x"].reshape(-1, D)
        grad_flat = grad.reshape(-1, D)
        dWo = x_flat.T @ grad_flat

        grad_r = grad.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        dV = cache["attn"].transpose(0, 1, 3, 2) @ grad_r
        dAttn = grad_r @ cache["V"].transpose(0, 1, 3, 2)
        dScores = dAttn * cache["attn"] * (1 - cache["attn"])

        dQ = dScores @ cache["K"]
        dK = dScores.transpose(0, 1, 3, 2) @ cache["Q"]

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, D)

        dx = dQ @ self.params["Wq"].T + dK @ self.params["Wk"].T + dV @ self.params["Wv"].T

        self.grads["Wq"] = x_flat.T @ dQ.reshape(-1, D)
        self.grads["Wk"] = x_flat.T @ dK.reshape(-1, D)
        self.grads["Wv"] = x_flat.T @ dV.reshape(-1, D)
        self.grads["Wo"] = dWo

        return dx

    def clear_cache(self):
        self._kv_cache = None

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def __repr__(self):
        return f"CausalSelfAttention(dim={self.embed_dim}, heads={self.num_heads}, rope={self.use_rope})"

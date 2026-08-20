import numpy as np
from layers.dense import Layer


class SelfAttention(Layer):
    """Multi-Head Self-Attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        scale = np.sqrt(2.0 / embed_dim)
        self.params["Wq"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wk"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wv"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wo"] = np.random.randn(embed_dim, embed_dim) * scale

        self.dropout_p = dropout

    def forward(self, x):
        B, T, D = x.shape

        Q = (x @ self.params["Wq"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.params["Wk"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.params["Wv"]).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)

        if self._training and self.dropout_p > 0:
            mask = (np.random.rand(*scores.shape) > self.dropout_p) / (1 - self.dropout_p)
            scores = scores * mask

        attn = self._softmax(scores)
        out = attn @ V

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = out @ self.params["Wo"]

        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "attn": attn, "B": B, "T": T}
        return out

    def backward(self, grad):
        B, T, D = grad.shape
        cache = self._cache

        x_flat = cache["x"].reshape(-1, D)
        grad_flat = grad.reshape(-1, D)
        dWo = x_flat.T @ grad_flat

        grad = grad.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        dV = cache["attn"].transpose(0, 1, 3, 2) @ grad
        dAttn = grad @ cache["V"].transpose(0, 1, 3, 2)
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

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def __repr__(self):
        return f"SelfAttention(dim={self.embed_dim}, heads={self.num_heads})"


class CrossAttention(Layer):
    """Multi-Head Cross-Attention (query de uma fonte, key/value de outra)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        scale = np.sqrt(2.0 / embed_dim)
        self.params["Wq"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wk"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wv"] = np.random.randn(embed_dim, embed_dim) * scale
        self.params["Wo"] = np.random.randn(embed_dim, embed_dim) * scale

    def forward(self, x, context):
        B, T_q, D = x.shape
        _, T_kv, _ = context.shape

        Q = (x @ self.params["Wq"]).reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (context @ self.params["Wk"]).reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (context @ self.params["Wv"]).reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        attn = self._softmax(scores)
        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T_q, D)
        out = out @ self.params["Wo"]

        self._cache = {"x": x, "context": context, "Q": Q, "K": K, "V": V, "attn": attn, "B": B, "T_q": T_q, "T_kv": T_kv}
        return out

    def backward(self, grad):
        cache = self._cache
        B, T_q, D = grad.shape

        dWo = cache["x"].reshape(-1, D).T @ grad.reshape(-1, D)
        grad_r = grad.reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        dV = cache["attn"].transpose(0, 1, 3, 2) @ grad_r
        dAttn = grad_r @ cache["V"].transpose(0, 1, 3, 2)
        dScores = dAttn * cache["attn"] * (1 - cache["attn"])

        dQ = dScores @ cache["K"]
        dK = dScores.transpose(0, 1, 3, 2) @ cache["Q"]

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T_q, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(-1, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(-1, D)

        dx = dQ @ self.params["Wq"].T
        dctx = dK @ self.params["Wk"].T + dV @ self.params["Wv"].T

        self.grads["Wq"] = cache["x"].reshape(-1, D).T @ dQ
        self.grads["Wk"] = cache["context"].reshape(-1, D).T @ dK.reshape(B * cache["T_kv"], D)
        self.grads["Wv"] = cache["context"].reshape(-1, D).T @ dV.reshape(B * cache["T_kv"], D)
        self.grads["Wo"] = dWo

        return dx, dctx

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def __repr__(self):
        return f"CrossAttention(dim={self.embed_dim}, heads={self.num_heads})"

import numpy as np
from ..layers.dense import Layer


class Embedding(Layer):
    """Camada de embedding — mapeia indices inteiros para vetores densos."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.params["W"] = np.random.randn(vocab_size, embed_dim) * 0.02

    def forward(self, x):
        self._cache["x_shape"] = x.shape
        self._cache["x"] = x
        return self.params["W"][x]

    def backward(self, grad):
        gradW = np.zeros_like(self.params["W"])
        x = self._cache["x"]
        np.add.at(gradW, x, grad)
        self.grads["W"] = gradW
        return None

    def __repr__(self):
        return f"Embedding({self.vocab_size}, {self.embed_dim})"

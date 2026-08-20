import numpy as np
from layers.dense import Layer


class RMSNorm(Layer):
    """Root Mean Square Normalization — mais rapido que LayerNorm.
    Usado em LLaMA, Mistral, Gemma."""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.params["weight"] = np.ones(num_features)

    def forward(self, x):
        self._cache["x"] = x
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.params["weight"]

    def backward(self, grad):
        x = self._cache["x"]
        weight = self.params["weight"]
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms

        self.grads["weight"] = (grad * x_norm).sum(axis=tuple(range(len(x.shape) - 1)))

        dx_norm = grad * weight
        drms = (-dx_norm * x / (rms ** 2)).sum(axis=-1, keepdims=True)
        dx = dx_norm / rms + 2.0 * x * drms / (x.shape[-1] * rms)
        return dx

    def __repr__(self):
        return f"RMSNorm({self.num_features})"

import numpy as np
from layers.dense import Layer


class LayerNorm(Layer):
    """Layer Normalization — normaliza por features, não por batch."""

    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.use_affine = affine

        if affine:
            self.params["gamma"] = np.ones(self.normalized_shape)
            self.params["beta"] = np.zeros(self.normalized_shape)

    def forward(self, x):
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)

        self._cache["x"] = x
        self._cache["mean"] = mean
        self._cache["var"] = var
        self._cache["axes"] = axes

        x_norm = (x - mean) / np.sqrt(var + self.eps)

        if self.use_affine:
            x_norm = x_norm * self.params["gamma"] + self.params["beta"]
        return x_norm

    def backward(self, grad):
        x = self._cache["x"]
        mean = self._cache["mean"]
        var = self._cache["var"]
        axes = self._cache["axes"]

        N = 1
        for a in axes:
            N *= x.shape[a]

        if self.use_affine:
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            self.grads["gamma"] = (grad * x_norm).sum(axis=axes)
            self.grads["beta"] = grad.sum(axis=axes)
            grad = grad * self.params["gamma"]

        dx_norm = grad
        dvar = (-0.5 * ((x - mean) * (var + self.eps) ** -1.5) * dx_norm).sum(
            axis=axes, keepdims=True
        )
        dmean = (
            (-1.0 / np.sqrt(var + self.eps)) * dx_norm.sum(axis=axes, keepdims=True)
            + dvar * (-2.0 / N) * (x - mean).sum(axis=axes, keepdims=True)
        )
        dx = (
            dx_norm / np.sqrt(var + self.eps)
            + dvar * 2.0 * (x - mean) / N
            + dmean / N
        )
        return dx

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"

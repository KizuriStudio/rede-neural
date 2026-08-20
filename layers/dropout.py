import numpy as np
from layers.dense import Layer


class Dropout(Layer):
    """Dropout aleatório com máscara binária invertida."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x):
        if self._training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self._mask
        return x

    def backward(self, grad):
        if self._training and self.p > 0:
            return grad * self._mask
        return grad

    def __repr__(self):
        return f"Dropout(p={self.p})"

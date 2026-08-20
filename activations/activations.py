import numpy as np
from layers.dense import Layer


class ReLU(Layer):
    def forward(self, x):
        self._cache = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self._cache > 0).astype(float)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self._cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self._cache > 0, 1.0, self.alpha)

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class GELU(Layer):
    """Gaussian Error Linear Unit — ativação usada em Transformers."""

    def forward(self, x):
        self._cache = x
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        return x * cdf

    def backward(self, grad):
        x = self._cache
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        sech2 = 1.0 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)) ** 2
        dcdf = 0.5 * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x ** 2)
        return grad * (cdf + x * dcdf)

    def __repr__(self):
        return "GELU()"


class Sigmoid(Layer):
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self._cache = 1.0 / (1.0 + np.exp(-x))
        return self._cache

    def backward(self, grad):
        return grad * self._cache * (1.0 - self._cache)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Layer):
    def forward(self, x):
        self._cache = np.tanh(x)
        return self._cache

    def backward(self, grad):
        return grad * (1.0 - self._cache ** 2)

    def __repr__(self):
        return "Tanh()"


class Softmax(Layer):
    def forward(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        self._cache = e / e.sum(axis=-1, keepdims=True)
        return self._cache

    def backward(self, grad):
        return self._cache * (grad - (grad * self._cache).sum(axis=-1, keepdims=True))

    def __repr__(self):
        return "Softmax()"


class ELU(Layer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self._cache = x
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    def backward(self, grad):
        x = self._cache
        return grad * np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -500, 500)))

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class Swish(Layer):
    """Swish = x * sigmoid(x) — usada no EfficientNet."""

    def forward(self, x):
        self._cache_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        self._cache_x = x
        return x * self._cache_sigmoid

    def backward(self, grad):
        x = self._cache_x
        sig = self._cache_sigmoid
        return grad * (sig + x * sig * (1 - sig))

    def __repr__(self):
        return "Swish()"

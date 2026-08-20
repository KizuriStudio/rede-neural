import numpy as np
from ..layers.dense import Layer


class BatchNorm1D(Layer):
    """Batch Normalization para dados 1D (batch, features)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.params["gamma"] = np.ones(num_features)
        self.params["beta"] = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self._training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        self._cache["x"] = x
        self._cache["mean"] = mean
        self._cache["var"] = var
        self._cache["N"] = x.shape[0]

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * self.params["gamma"] + self.params["beta"]

    def backward(self, grad):
        x = self._cache["x"]
        mean = self._cache["mean"]
        var = self._cache["var"]
        N = self._cache["N"]

        x_norm = (x - mean) / np.sqrt(var + self.eps)

        self.grads["gamma"] = (grad * x_norm).sum(axis=0)
        self.grads["beta"] = grad.sum(axis=0)

        grad = grad * self.params["gamma"]
        dvar = (-0.5 * ((x - mean) * (var + self.eps) ** -1.5) * grad).sum(axis=0)
        dmean = (-1.0 / np.sqrt(var + self.eps)) * grad.sum(axis=0) + dvar * (-2.0 / N) * (x - mean).sum(axis=0)

        dx = grad / np.sqrt(var + self.eps) + dvar * 2.0 * (x - mean) / N + dmean / N
        return dx

    def __repr__(self):
        return f"BatchNorm1D({self.num_features})"


class BatchNorm2D(Layer):
    """Batch Normalization para dados 2D (batch, channels, h, w)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.params["gamma"] = np.ones(num_features)
        self.params["beta"] = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        N, C, H, W = x.shape
        if self._training:
            mean = x.reshape(N, C, -1).mean(axis=(0, 2))
            var = x.reshape(N, C, -1).var(axis=(0, 2))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        self._cache = {"x": x, "mean": mean, "var": var, "N": N}

        x_norm = (x - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + self.eps)
        return x_norm * self.params["gamma"][None, :, None, None] + self.params["beta"][None, :, None, None]

    def backward(self, grad):
        x = self._cache["x"]
        mean = self._cache["mean"]
        var = self._cache["var"]
        N = self._cache["N"]
        _, C, H, W = x.shape

        x_norm = (x - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + self.eps)
        self.grads["gamma"] = (grad * x_norm).sum(axis=(0, 2, 3))
        self.grads["beta"] = grad.sum(axis=(0, 2, 3))

        grad = grad * self.params["gamma"][None, :, None, None]
        dvar = (-0.5 * ((x - mean[None, :, None, None]) * (var[None, :, None, None] + self.eps) ** -1.5) * grad).sum(axis=(0, 2, 3))
        dmean = (-1.0 / np.sqrt(var[None, :, None, None] + self.eps)) * grad.sum(axis=(0, 2, 3)) + dvar * (-2.0 / (N * H * W)) * (x - mean[None, :, None, None]).sum(axis=(0, 2, 3))

        dx = grad / np.sqrt(var[None, :, None, None] + self.eps) + dvar[None, :, None, None] * 2.0 * (x - mean[None, :, None, None]) / (N * H * W) + dmean[None, :, None, None] / (N * H * W)
        return dx

    def __repr__(self):
        return f"BatchNorm2D({self.num_features})"

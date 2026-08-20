import numpy as np


class Layer:
    base_class = True

    def __init__(self):
        self.params = {}
        self.grads = {}
        self._training = True
        self._name = self.__class__.__name__
        self._cache = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def parameters(self):
        return self.params

    def gradients(self):
        return self.grads

    def __repr__(self):
        return f"{self._name}()"


class Dense(Layer):
    """Camada fully connected com suporte a bias."""

    def __init__(self, in_features, out_features, bias=True, weight_init="he"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if weight_init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif weight_init == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        elif weight_init == "lecun":
            scale = np.sqrt(1.0 / in_features)
        else:
            scale = 0.01

        self.params["W"] = np.random.randn(in_features, out_features) * scale
        if bias:
            self.params["b"] = np.zeros(out_features)

        self._cache = {}

    def forward(self, x):
        self._cache["x"] = x
        out = x @ self.params["W"]
        if self.use_bias:
            out += self.params["b"]
        return out

    def backward(self, grad):
        x = self._cache["x"]
        x2d = x.reshape(-1, x.shape[-1])
        grad2d = grad.reshape(-1, grad.shape[-1])
        self.grads["W"] = x2d.T @ grad2d
        if self.use_bias:
            self.grads["b"] = grad2d.sum(axis=0)
        return (grad2d @ self.params["W"].T).reshape(x.shape)

    def __repr__(self):
        return f"Dense({self.in_features} -> {self.out_features}, bias={self.use_bias})"

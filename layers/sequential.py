import numpy as np
from layers.dense import Layer


class Sequential(Layer):
    """Container para empilhar camadas em sequência."""

    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers or []
        self._build_params()

    def _build_params(self):
        self.params = {}
        self.grads = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                self.params[f"{i}.{k}"] = v
            for k, v in layer.grads.items():
                self.grads[f"{i}.{k}"] = v

    def add(self, layer):
        self.layers.append(layer)
        self._build_params()

    def forward(self, x):
        self._cache = {"x": [x]}
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                self._cache["x"].append(x)
        return x

    def backward(self, grad):
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                x = self._cache["x"][i + 1]
            grad = self.layers[i].backward(grad)
        self._sync_grads()
        return grad

    def _sync_grads(self):
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads.items():
                self.grads[f"{i}.{k}"] = v

    def train(self):
        self._training = True
        for l in self.layers:
            l.train()

    def eval(self):
        self._training = False
        for l in self.layers:
            l.eval()

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                params[f"{i}.{k}"] = v
        return params

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        lines = ["Sequential("]
        for i, l in enumerate(self.layers):
            lines.append(f"  ({i}): {l}")
        lines.append(")")
        return "\n".join(lines)

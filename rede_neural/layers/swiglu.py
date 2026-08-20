import numpy as np
from ..layers.dense import Layer


class SwiGLU(Layer):
    """SwiGLU = Swish(xW1) * (xW2) — ativação usada em LLaMA, PaLM."""

    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        scale = np.sqrt(2.0 / embed_dim)
        self.params["W1"] = np.random.randn(embed_dim, ffn_dim) * scale
        self.params["W2"] = np.random.randn(embed_dim, ffn_dim) * scale
        self.params["W3"] = np.random.randn(ffn_dim, embed_dim) * scale

    def _silu(self, x):
        return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        self._cache["x"] = x
        gate = x @ self.params["W1"]
        up = x @ self.params["W2"]
        swish = self._silu(gate)
        gate_act = swish * up
        out = gate_act @ self.params["W3"]
        self._cache["gate"] = gate
        self._cache["up"] = up
        self._cache["swish"] = swish
        self._cache["gate_act"] = gate_act
        return out

    def backward(self, grad):
        x = self._cache["x"]
        gate = self._cache["gate"]
        up = self._cache["up"]
        swish = self._cache["swish"]
        gate_act = self._cache["gate_act"]

        x2d = x.reshape(-1, x.shape[-1])
        grad2d = grad.reshape(-1, grad.shape[-1])
        gate_act2d = gate_act.reshape(-1, gate_act.shape[-1])
        gate2d = gate.reshape(-1, gate.shape[-1])
        up2d = up.reshape(-1, up.shape[-1])
        swish2d = swish.reshape(-1, swish.shape[-1])

        self.grads["W3"] = gate_act2d.T @ grad2d
        d_gate_act = grad2d @ self.params["W3"].T

        d_swish = d_gate_act * up2d
        d_up = d_gate_act * swish2d

        sig = 1.0 / (1.0 + np.exp(-np.clip(gate2d, -500, 500)))
        d_gate = d_swish * (sig + gate2d * sig * (1 - sig))

        dx = (d_gate @ self.params["W1"].T + d_up @ self.params["W2"].T).reshape(x.shape)

        self.grads["W1"] = x2d.T @ d_gate
        self.grads["W2"] = x2d.T @ d_up

        return dx

    def __repr__(self):
        return f"SwiGLU({self.embed_dim} -> {self.ffn_dim})"

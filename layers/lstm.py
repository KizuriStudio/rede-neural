import numpy as np
from layers.dense import Layer


class LSTM(Layer):
    """LSTM com forget gate, input gate, output gate e cell state."""

    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        total_in = input_size + hidden_size
        scale = np.sqrt(2.0 / total_in)

        self.params["Wf"] = np.random.randn(total_in, hidden_size) * scale
        self.params["Wi"] = np.random.randn(total_in, hidden_size) * scale
        self.params["Wc"] = np.random.randn(total_in, hidden_size) * scale
        self.params["Wo"] = np.random.randn(total_in, hidden_size) * scale

        self.params["bf"] = np.zeros(hidden_size)
        self.params["bi"] = np.zeros(hidden_size)
        self.params["bc"] = np.zeros(hidden_size)
        self.params["bo"] = np.zeros(hidden_size)

        for k in self.params:
            self.grads[k] = np.zeros_like(self.params[k])

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _step(self, x_t, h_prev, c_prev):
        combined = np.concatenate([h_prev, x_t], axis=-1)

        f = self._sigmoid(combined @ self.params["Wf"] + self.params["bf"])
        i = self._sigmoid(combined @ self.params["Wi"] + self.params["bi"])
        c_tilde = np.tanh(combined @ self.params["Wc"] + self.params["bc"])
        o = self._sigmoid(combined @ self.params["Wo"] + self.params["bo"])

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        return h, c, {"f": f, "i": i, "c_tilde": c_tilde, "o": o, "combined": combined}

    def forward(self, x):
        if x.ndim == 2:
            x = x[np.newaxis, :, :]

        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))

        self._cache = {
            "x": x, "h": [h.copy()], "c": [c.copy()],
            "gates": [], "seq_len": seq_len, "batch": batch
        }

        outputs = []
        for t in range(seq_len):
            h, c, gates = self._step(x[:, t, :], h, c)
            outputs.append(h)
            self._cache["h"].append(h.copy())
            self._cache["c"].append(c.copy())
            self._cache["gates"].append(gates)

        self._cache["outputs"] = np.stack(outputs, axis=1)
        return self._cache["outputs"]

    def backward(self, grad):
        if grad.ndim == 2:
            grad = grad[np.newaxis, :, :]

        x = self._cache["x"]
        seq_len = self._cache["seq_len"]
        batch = self._cache["batch"]

        dWf = np.zeros_like(self.params["Wf"])
        dWi = np.zeros_like(self.params["Wi"])
        dWc = np.zeros_like(self.params["Wc"])
        dWo = np.zeros_like(self.params["Wo"])
        dbf = np.zeros_like(self.params["bf"])
        dbi = np.zeros_like(self.params["bi"])
        dbc = np.zeros_like(self.params["bc"])
        dbo = np.zeros_like(self.params["bo"])

        dh_next = np.zeros((batch, self.hidden_size))
        dc_next = np.zeros((batch, self.hidden_size))

        dx_list = []

        for t in reversed(range(seq_len)):
            dh = grad[:, t, :] + dh_next
            dc = dc_next

            c = self._cache["c"][t + 1]
            c_prev = self._cache["c"][t]
            gates = self._cache["gates"][t]

            o = gates["o"]
            i = gates["i"]
            f = gates["f"]
            c_tilde = gates["c_tilde"]
            combined = gates["combined"]

            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc += dh * o * (1 - tanh_c ** 2)

            di = dc * c_tilde
            dc_tilde = dc * i
            df = dc * c_prev
            dc_prev = dc * f

            d_c_tilde = dc_tilde * (1 - c_tilde ** 2)

            dcombined_f = df * f * (1 - f)
            dcombined_i = di * i * (1 - i)
            dcombined_c = d_c_tilde
            dcombined_o = do * o * (1 - o)

            dWf += combined.T @ dcombined_f
            dWi += combined.T @ dcombined_i
            dWc += combined.T @ dcombined_c
            dWo += combined.T @ dcombined_o

            dbf += dcombined_f.sum(axis=0)
            dbi += dcombined_i.sum(axis=0)
            dbc += dcombined_c.sum(axis=0)
            dbo += dcombined_o.sum(axis=0)

            dcombined = (
                dcombined_f @ self.params["Wf"].T
                + dcombined_i @ self.params["Wi"].T
                + dcombined_c @ self.params["Wc"].T
                + dcombined_o @ self.params["Wo"].T
            )

            dx_t = dcombined[:, self.hidden_size:]
            dh_next = dcombined[:, :self.hidden_size]
            dc_next = dc_prev

            dx_list.append(dx_t)

        dx = np.stack(dx_list[::-1], axis=1)

        self.grads["Wf"] = dWf
        self.grads["Wi"] = dWi
        self.grads["Wc"] = dWc
        self.grads["Wo"] = dWo
        self.grads["bf"] = dbf
        self.grads["bi"] = dbi
        self.grads["bc"] = dbc
        self.grads["bo"] = dbo

        return dx

    def __repr__(self):
        return f"LSTM({self.input_size} -> {self.hidden_size}, bidir={self.bidirectional})"


class BiLSTM(Layer):
    """LSTM bidirecional — processa a sequência nos dois sentidos."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fwd_lstm = LSTM(input_size, hidden_size)
        self.bwd_lstm = LSTM(input_size, hidden_size)

        self.params = {
            "Wf": self.fwd_lstm.params["Wf"],
            "Wi": self.fwd_lstm.params["Wi"],
            "Wc": self.fwd_lstm.params["Wc"],
            "Wo": self.fwd_lstm.params["Wo"],
            "bf": self.fwd_lstm.params["bf"],
            "bi": self.fwd_lstm.params["bi"],
            "bc": self.fwd_lstm.params["bc"],
            "bo": self.fwd_lstm.params["bo"],
            "Wf_b": self.bwd_lstm.params["Wf"],
            "Wi_b": self.bwd_lstm.params["Wi"],
            "Wc_b": self.bwd_lstm.params["Wc"],
            "Wo_b": self.bwd_lstm.params["Wo"],
            "bf_b": self.bwd_lstm.params["bf"],
            "bi_b": self.bwd_lstm.params["bi"],
            "bc_b": self.bwd_lstm.params["bc"],
            "bo_b": self.bwd_lstm.params["bo"],
        }

    def forward(self, x):
        fwd_out = self.fwd_lstm.forward(x)
        bwd_out = self.bwd_lstm.forward(x[:, ::-1, :])
        bwd_out = bwd_out[:, ::-1, :]
        self._cache = {"fwd": fwd_out, "bwd": bwd_out}
        return np.concatenate([fwd_out, bwd_out], axis=-1)

    def backward(self, grad):
        grad_fwd = grad[:, :, :self.hidden_size]
        grad_bwd = grad[:, :, self.hidden_size:]

        dx_fwd = self.fwd_lstm.backward(grad_fwd)
        dx_bwd = self.bwd_lstm.backward(grad_bwd[:, ::-1, :])[:, ::-1, :]

        self.grads = {
            "Wf": self.fwd_lstm.grads["Wf"],
            "Wi": self.fwd_lstm.grads["Wi"],
            "Wc": self.fwd_lstm.grads["Wc"],
            "Wo": self.fwd_lstm.grads["Wo"],
            "bf": self.fwd_lstm.grads["bf"],
            "bi": self.fwd_lstm.grads["bi"],
            "bc": self.fwd_lstm.grads["bc"],
            "bo": self.fwd_lstm.grads["bo"],
            "Wf_b": self.bwd_lstm.grads["Wf"],
            "Wi_b": self.bwd_lstm.grads["Wi"],
            "Wc_b": self.bwd_lstm.grads["Wc"],
            "Wo_b": self.bwd_lstm.grads["Wo"],
            "bf_b": self.bwd_lstm.grads["bf"],
            "bi_b": self.bwd_lstm.grads["bi"],
            "bc_b": self.bwd_lstm.grads["bc"],
            "bo_b": self.bwd_lstm.grads["bo"],
        }

        return dx_fwd + dx_bwd

    def __repr__(self):
        return f"BiLSTM({self.input_size} -> {self.hidden_size})"

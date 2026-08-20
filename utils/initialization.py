import numpy as np


def he_init(shape, activation="relu"):
    """He initialization — ideal pra ReLU."""
    fan_in = shape[0]
    if activation == "relu":
        std = np.sqrt(2.0 / fan_in)
    elif activation == "leaky_relu":
        std = np.sqrt(2.0 / (1.0 + 0.01 ** 2) / fan_in)
    else:
        std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


def xavier_init(shape):
    """Xavier/Glorot initialization — ideal pra tanh/sigmoid."""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def orthogonal_init(shape):
    """Orthogonal initialization — preserva gradientes em RNNs/LSTMs."""
    flat_shape = (shape[0], np.prod(shape[1:]))
    num_rows, num_cols = flat_shape
    num_repeats = min(num_rows, num_cols)

    flat = np.random.randn(flat_shape[0], flat_shape[1])
    if num_rows < num_cols:
        flat = flat.T

    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))

    if num_rows < num_cols:
        q = q.T

    return q.reshape(shape)


def kaiming_init(shape, activation="relu"):
    """Alias pra he_init."""
    return he_init(shape, activation)

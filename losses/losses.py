import numpy as np


class Loss:
    def forward(self, predictions, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Cross-Entropy com softmax embutido (mais estável numericamente)."""

    def forward(self, logits, targets):
        self._logits = logits
        self._targets = targets

        batch_size = logits.shape[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp

        if targets.ndim == 1:
            self._log_probs = log_probs
            loss = -log_probs[np.arange(batch_size), targets].mean()
        else:
            self._log_probs = log_probs
            loss = -(log_probs * targets).sum(axis=-1).mean()

        return loss

    def backward(self):
        batch_size = self._logits.shape[0]
        shifted = self._logits - self._logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        if self._targets.ndim == 1:
            grad = probs.copy()
            grad[np.arange(batch_size), self._targets] -= 1
            grad /= batch_size
        else:
            grad = (probs - self._targets) / batch_size

        return grad


class MSELoss(Loss):
    def forward(self, predictions, targets):
        self._diff = predictions - targets
        return (self._diff ** 2).mean()

    def backward(self):
        return 2.0 * self._diff / self._diff.size


class BCELoss(Loss):
    def forward(self, predictions, targets):
        self._pred = np.clip(predictions, 1e-7, 1 - 1e-7)
        self._target = targets
        return -(targets * np.log(self._pred) + (1 - targets) * np.log(1 - self._pred)).mean()

    def backward(self):
        batch = self._pred.shape[0]
        return (-(self._target / self._pred) + (1 - self._target) / (1 - self._pred)) / batch


class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, predictions, targets):
        self._diff = predictions - targets
        abs_diff = np.abs(self._diff)
        self._mask = abs_diff <= self.delta
        loss = np.where(
            self._mask,
            0.5 * self._diff ** 2,
            self.delta * abs_diff - 0.5 * self.delta ** 2
        )
        return loss.mean()

    def backward(self):
        return np.where(
            self._mask,
            self._diff,
            self.delta * np.sign(self._diff)
        ) / self._diff.size


class LabelSmoothingCE(Loss):
    def __init__(self, classes, smoothing=0.1):
        self.classes = classes
        self.smoothing = smoothing
        self.ce = CrossEntropyLoss()

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        smooth = np.full((batch_size, self.classes), self.smoothing / (self.classes - 1))
        smooth[np.arange(batch_size), targets] = 1.0 - self.smoothing
        return self.ce.forward(logits, smooth)

    def backward(self):
        return self.ce.backward()

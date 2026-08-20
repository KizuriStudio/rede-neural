import numpy as np


class Optimizer:
    def __init__(self, params, grads, lr):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.t = 0

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for k in self.grads:
            self.grads[k] = np.zeros_like(self.grads[k])


class SGD(Optimizer):
    def __init__(self, params, grads, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, grads, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self):
        self.t += 1
        for k in self.params:
            g = self.grads[k]
            if self.weight_decay > 0:
                g = g + self.weight_decay * self.params[k]

            self.velocities[k] = self.momentum * self.velocities[k] + self.lr * g

            if self.nesterov:
                self.params[k] -= self.momentum * self.velocities[k] + self.lr * g
            else:
                self.params[k] -= self.velocities[k]


class Adam(Optimizer):
    def __init__(self, params, grads, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, grads, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self):
        self.t += 1
        for k in self.params:
            g = self.grads[k]
            if self.weight_decay > 0:
                g = g + self.weight_decay * self.params[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Adam com weight decay decoupled — melhor que Adam pra generalização."""

    def __init__(self, params, grads, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, grads, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self):
        self.t += 1
        for k in self.params:
            g = self.grads[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            self.params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.params[k])


class RAdam(Optimizer):
    """Rectified Adam — Adam com variance correction mais estável no início."""

    def __init__(self, params, grads, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, grads, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self):
        self.t += 1
        rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
        rho_t = rho_inf - 2.0 * self.t * (self.beta2 ** self.t) / (1.0 - self.beta2 ** self.t)

        for k in self.params:
            g = self.grads[k]
            if self.weight_decay > 0:
                g = g + self.weight_decay * self.params[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            if rho_t > 5:
                l_t = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                self.params[k] -= self.lr * l_t * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                self.params[k] -= self.lr * m_hat


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments — bom pra batch grande."""

    def __init__(self, params, grads, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, grads, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self):
        self.t += 1
        for k in self.params:
            g = self.grads[k]

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            r = m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.params[k]

            norm_w = np.linalg.norm(self.params[k])
            norm_r = np.linalg.norm(r)

            if norm_w > 0 and norm_r > 0:
                self.params[k] -= self.lr * norm_w / norm_r * r
            else:
                self.params[k] -= self.lr * r

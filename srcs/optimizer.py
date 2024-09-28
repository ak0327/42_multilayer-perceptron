import numpy as np


class SGD:
    """
    θ(t+1) = θ(t) - η * ∂L/∂θ(t)
    """
    def __init__(self, lr: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    θ(t+1) = θ(t) + v(t+1)
    v(t+1) = α * v(t) - η * ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            momentum: float = 0.9
    ):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params:
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] -= self.v[key]


class Nesterov:
    """
    http://arxiv.org/abs/1212.0901
    """
    def __init__(
            self,
            lr: float = 0.01,
            momentum: float = 0.9
    ):
        self.lr = lr
        self.momentum = momentum
        self. v = None

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            v_prev = self.v[key].copy()
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] -= self.momentum * v_prev
            params[key] += (1 + self.momentum) * self.v[key]


class AdaGrad:
    """
    θ(t+1) = θ(t) - η / sqrt(h(t+1)) * ∂L/∂θ(t)
    h(t+1) = h(t) + ∂L/∂θ(t) @ ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            epsilon: float = 1e-7
    ):
        self.lr = lr
        self.h = None
        self.epsilon = epsilon

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            sqrt_h = np.sqrt(self.h[key]) + self.epsilon
            params[key] -= self.lr / sqrt_h * grads[key]


class RMSProp:
    """
    θ(t+1) = θ(t) - η / sqrt(h(t+1)) * ∂L/∂θ(t)
    h(t+1) = ρ * h(t) + (1 - ρ) * ∂L/∂θ(t) @ ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            decay_rate: float = 0.99,
            epsilon: float = 1e-7
    ):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        self.epsilon = epsilon

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] = (1 - self.decay_rate) * grads[key] * grads[key]
            sqrt_h = np.sqrt(self.h[key]) + self.epsilon
            params[key] -= self.lr / sqrt_h * grads[key]


class Adam:
    """
    θ(t+1)     = θ(t) - η / sqrt(v_hat(t+1)) * m_hat(t+1)
    m_hat(t+1) = m(t+1) / (1 - β1 ** t)
    v_hat(t+1) = v(t+1) / (1 - β2 ** t)
    m(t+1)     = β1 * m(t) + (1 - β1) * ∂L/∂θ(t)
    v(t+1)     = β2 * v(t) + (1 - β2) * ∂L/∂θ(t) @ ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-7
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.epsilon = epsilon

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        for key in params.keys():
            self.m[key] *= self.beta1
            self.m[key] += (1 - self.beta1) * grads[key]
            self.v[key] *= self.beta2
            self.v[key] += (1 - self.beta2) * grads[key] * grads[key]
            m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
            v_hat = self.v[key] / (1 - self.beta2 ** self.iter)
            sqrt_v_hat = np.sqrt(v_hat) + self.epsilon
            params[key] -= self.lr / sqrt_v_hat * m_hat

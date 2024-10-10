import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        pass


class SGD(Optimizer):
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


class Momentum(Optimizer):
    """
    θ(t+1) = θ(t) + v(t+1)
    v(t+1) = α * v(t) - η * ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            momentum: float = 0.9,
            nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        self.lr = lr
        self.momentum = momentum

        if nesterov:
            self.optimizer = Nesterov(lr=lr, momentum=momentum)
        else:
            self.optimizer = self

        self.v = None

    def update(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
    ) -> None:
        if isinstance(self.optimizer, Nesterov):
            self.optimizer.update(params, grads)
        else:
            if self.v is None:
                self.v = {}
                for key, val in params.items():
                    self.v[key] = np.zeros_like(val)

            for key in params.keys():
                if np.isnan(self.momentum) or np.isinf(self.momentum):
                    self.v[key] = -self.lr * grads[key]
                else:
                    self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]


class Nesterov(Optimizer):
    """
    http://arxiv.org/abs/1212.0901
    """
    def __init__(
            self,
            lr: float = 0.01,
            momentum: float = 0.9
    ):
        if momentum == 0.0:
            raise ValueError(f"Nesterov momentum requires a momentum and zero dampening")

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


class AdaGrad(Optimizer):
    """
    θ(t+1) = θ(t) - η / sqrt(h(t+1)) * ∂L/∂θ(t)
    h(t+1) = h(t) + ∂L/∂θ(t) @ ∂L/∂θ(t)
    """
    def __init__(
            self,
            lr: float = 0.01,
            epsilon: float = 1e-10
    ):
        if lr < 0.0 or np.isnan(lr):
            raise ValueError(f"Invalid learning rate: {lr}")

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
            self.h[key] += grads[key] ** 2
            adjusted_lr = self.lr / (np.sqrt(self.h[key]) + self.epsilon)
            params[key] -= adjusted_lr * grads[key]


class RMSProp(Optimizer):
    """
    θ(t+1) = θ(t) - η / sqrt(h(t+1)) * ∂L/∂θ(t)
    h(t+1) = ρ * h(t) + (1 - ρ) * ( ∂L/∂θ(t) )^2
    """
    def __init__(
            self,
            lr: float = 0.01,
            alpha: float = 0.99,
            epsilon: float = 1e-8
    ):
        if lr < 0.0 or np.isnan(lr):
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0.0 <= alpha < 1.0) or np.isnan(alpha):
            raise ValueError(f"Invalid alpha value: {alpha}. Must be in [0, 1).")

        self.lr = lr
        self.alpha = alpha
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
            self.h[key] *= self.alpha
            self.h[key] += (1 - self.alpha) * grads[key] ** 2
            adjusted_lr = self.lr / (np.sqrt(self.h[key]) + self.epsilon)
            params[key] -= adjusted_lr * grads[key]


class Adam(Optimizer):
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
            betas: tuple[float, float] = (0.9, 0.999),
            epsilon: float = 1e-8
    ):
        if lr < 0.0 or np.isnan(lr):
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2:
            raise ValueError(f"Invalid beta parameter: Must be a two floats")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
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

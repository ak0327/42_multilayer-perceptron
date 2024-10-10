import numpy as np


class ReLU():
    def __init__(self):
        self.mask = None

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        """
        out = 0 (x <= 0)
        out = x (0 < x)
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray):
        """
        dx = 0    (x <= 0)
        dx = dout (0 < x)
        """
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

    @property
    def info(self):
        return f"ReLU()"


class Sigmoid():
    def __init__(self):
        self.out = None

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        """
        return 1 / (1 + np.exp(-x))
        x < 0 の時、exp(-x) = inf -> sigmoid(x) = nanになる可能性がある
        x < 0  : sigmoid(x) = exp(x) / (1 + np.exp(x))
        0 <= x : sigmoid(x) = 1 / (1 + np.exp(-x))
        """
        out = np.exp(np.minimum(0, x)) / (1 + np.exp(-np.abs(x)))
        self.out = out
        return out

    def backward(self, dout: np.ndarray):
        dx = dout * (1.0 - self.out) * self.out
        return dx

    @property
    def info(self):
        return f"Sigmoid()"

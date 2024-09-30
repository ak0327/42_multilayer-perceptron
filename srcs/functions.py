import numpy as np


class Softmax:
    def __call__(self, x: np.ndarray):
        if x.size == 0:
            return x

        if np.isinf(x).any() or np.isnan(x).any():
            return np.full_like(x, np.nan)

        x -= x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray):
        return self(x) * (1 - self(x))

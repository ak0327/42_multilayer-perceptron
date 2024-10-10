import numpy as np


class Softmax:
    def __init__(self):
        self.out = None  # Softmaxの出力を保持

    def __call__(self, x: np.ndarray):
        if x.size == 0:
            return x

        if np.isinf(x).any() or np.isnan(x).any():
            return np.full_like(x, np.nan)

        x -= x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x)
        self.out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.out

    def backward(self, delta: np.ndarray):
        # deltaはクロスエントロピー損失からの勾配
        batch_size = delta.shape[0]
        dx = self.out - delta  # self.out = y、delta = t の場合
        dx /= batch_size
        return dx

    @property
    def info(self):
        return f"Softmax()"

def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))

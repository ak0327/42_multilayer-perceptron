import numpy as np
from srcs.loss import cross_entropy


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


class SoftmaxWithCrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def __call__(self, x, t):
        return self.forward(x, t)

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = cross_entropy(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    @property
    def info(self):
        return f"SoftmaxWithCrossEntropyLoss()"


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))

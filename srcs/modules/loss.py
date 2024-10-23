import numpy as np
import sys


class CrossEntropyLoss:
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        self.y = None
        self.t = None

    def __call__(self, y: np.ndarray, t: np.ndarray) -> float:
        return self.forward(y, t)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if y.size == t.size:
            t = t.argmax(axis=1)
        t = t.astype(int)

        self.y = y
        self.t = t
        batch_size = y.shape[0]

        loss = -np.sum(np.log(y[np.arange(batch_size), t] + self.epsilon)) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        batch_size = self.y.shape[0]

        if self.t.size == self.y.size:  # 教師データがone-hot vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx

    @property
    def info(self):
        return f"CrossEntropyLoss()"


class BinaryCrossEntropyLoss:
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        self.y = None
        self.t = None
        self.t_one_hot = None

    def __call__(self, y: np.ndarray, t: np.ndarray) -> float:
        return self.forward(y, t)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = y
        self.t = t

        if t.ndim == 1:
            batch_size = t.size
            self.t_one_hot = np.zeros((batch_size, y.shape[1]))
            self.t_one_hot[np.arange(batch_size), t] = 1
        else:
            self.t_one_hot = t

        # クリッピングを適用
        y_clipped = np.clip(y, self.epsilon, 1.0 - self.epsilon)

        # バッチ当たりの平均損失を計算
        batch_size = y.shape[0]
        loss = -np.sum(self.t_one_hot * np.log(y_clipped) +
                       (1 - self.t_one_hot) * np.log(1 - y_clipped)) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        y_clipped = np.clip(self.y, self.epsilon, 1.0 - self.epsilon)
        batch_size = self.y.shape[0]
        grad = (y_clipped - self.t_one_hot) / (y_clipped * (1 - y_clipped) * batch_size)
        return grad

    @property
    def info(self):
        return f"BinaryCrossEntropyLoss()"

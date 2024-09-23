import numpy as np


def cross_entropy(
        y: np.ndarray,
        t: np.ndarray,
        epsilon: float = 1e-7
):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + epsilon)) / batch_size

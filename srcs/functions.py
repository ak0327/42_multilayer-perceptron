import numpy as np


def softmax(x):
    if x.size == 0:
        return x

    if np.isinf(x).any() or np.isnan(x).any():
        return np.full_like(x, np.nan)

    x -= x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

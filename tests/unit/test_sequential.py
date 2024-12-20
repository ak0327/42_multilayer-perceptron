import os
# import sys
# sys.path.append(os.pardir)

import numpy as np
from srcs.modules.functions import Softmax
from srcs.modules.activation import ReLU, Sigmoid
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.init import he_normal, xavier_normal, normal
from srcs.modules.optimizer import (
    SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam)
from srcs.modules.layer import Dense
from srcs.modules.model import Sequential
from srcs.train import create_model

from sklearn.model_selection import train_test_split
from keras.datasets import mnist


np.random.seed(42)


def get_mnist():
    (x_mnist_1, t_mnist_1), (x_mnist_2, t_mnist_2) = mnist.load_data()

    x_mnist = np.r_[x_mnist_1, x_mnist_2]
    t_mnist = np.r_[t_mnist_1, t_mnist_2]

    x_mnist = x_mnist.astype("float64") / 255.  # 値を[0, 1]に正規化する
    t_mnist = np.eye(N=10)[t_mnist.astype("int32").flatten()]  # one-hotベクトルにする

    x_mnist = x_mnist.reshape(x_mnist.shape[0], -1)  # 1次元に変換

    # train data: 5000, valid data: 10000 test data: 10000にする
    x_train, x_test, t_train, t_test = train_test_split(x_mnist, t_mnist, test_size=10000)
    x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=10000)

    print(f"x_train.shape: {x_train.shape}")

    print(f"t_train.shape: {t_train.shape}\n")
    return x_train, x_valid, t_train, t_valid


def normed_relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))


def _test_gradient(
        model: Sequential,
        x_train: np.ndarray,
        t_train: np.ndarray,
        batch_size: int
):
    print(f"Testing Gradient...")

    x = x_train[:batch_size]
    t = t_train[:batch_size]

    is_ok = True
    GRADIENT_TOLERANCE = 1e-5

    # 数値微分と誤差逆伝播法で勾配算出
    grad_numerical = model.numerical_grad(x, t)
    # print(f"grad_numerical: {grad_numerical}")

    y = model.forward(x)
    _ = model.loss(y, t)
    grad_backprop = model.backward()
    # print(f"grad_backprop: {grad_backprop}")

    # 各重みの差を確認
    for key in grad_numerical.keys():
        _grads_bprop = grad_backprop[key]
        _grads_num = grad_numerical[key]

        abs_diff = np.abs(_grads_bprop - _grads_num)
        relative_diff = normed_relative_diff(_grads_bprop, _grads_num)

        print(f"{key}: [abs_diff] {np.average(abs_diff):.5e}  [relative_diff] {relative_diff:.5e}")
        is_ok &= relative_diff < GRADIENT_TOLERANCE

    assert is_ok, \
        "[Failed] Gradient check, " \
        "The relative difference between numerical gradient and " \
        "backpropagation gradient exceeds the tolerance threshold."

    print(f"Test Gradient: PASS\n")


def test_sequential_grad():
    x_train, x_valid, t_train, t_valid = get_mnist()

    optimizers = ["SGD", "MOMENTUM", "NESTEROV", "ADAGRAD", "RMSPROP", "ADAM"]
    for optimizer in optimizers:
        net = create_model(
            features=[784, 10, 10],
            learning_rate=0.01,
            optimp_str=optimizer
        )
        _test_gradient(model=net, x_train=x_train, t_train=t_train, batch_size=3)

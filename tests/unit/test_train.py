import os
import sys
sys.path.append(os.pardir)

import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from tqdm import tqdm

from srcs.modules.functions import Softmax, np_log, numerical_gradient
from srcs.modules.activation import ReLU, Sigmoid
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.init import he_normal, xavier_normal, normal
from srcs.modules.optimizer import Optimizer
from srcs.modules.optimizer import (
    SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
)
from srcs.modules.layer import Dense
from srcs.modules.model import Sequential
from srcs.dataloader import get_wdbc
from srcs.train import train_model, create_model, _get_train_data


np.random.seed(34)


def normed_relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))


def _get_mnist():
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


def test_wdbc():
    seed = 42

    X_train, X_valid, t_train, t_valid = _get_train_data(dataset_path="data/data.csv")

    net = create_model(
        features=[30, 50, 50, 2],
        learning_rate=0.001,
        optimp_str="ADAM"
    )

    _, _, train_accs, _, valid_accs = train_model(
        model=net,
        X_train=X_train,
        t_train=t_train,
        X_valid=X_valid,
        t_valid=t_valid,
        iters_num=2500,
        verbose=False,
        plot=False,
        metrics_interval=100,
        name="WDBC"
    )
    assert 0.93 <= train_accs[-1]
    assert 0.93 <= valid_accs[-1]


def test_mnist():
    seed = 42

    X_train, X_valid, t_train, t_valid = _get_mnist()

    # lr = 0.001
    # optimizer = Adam(lr=lr)
    # net = Sequential(
    #     layers=[
    #         Dense(in_features=784, out_features=50, activation=ReLU, init_method=he_normal, seed=seed),
    #         Dense(in_features=50, out_features=10, activation=Softmax, init_method=xavier_normal, seed=seed)
    #     ],
    #     criteria=CrossEntropyLoss,
    #     optimizer=optimizer,
    # )
    net = create_model(
        features=[784, 50, 10],
        learning_rate=0.001,
        optimp_str="ADAM"
    )

    _, _, train_accs, _, valid_accs = train_model(
        model=net,
        X_train=X_train,
        t_train=t_train,
        X_valid=X_valid,
        t_valid=t_valid,
        iters_num=10,
        verbose=True,
        plot=False,
        name="MNIST"
    )
    assert 0.93 <= train_accs[-1]
    assert 0.93 <= valid_accs[-1]

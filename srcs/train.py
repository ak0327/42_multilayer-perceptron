import os
import sys
sys.path.append(os.pardir)

import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class RealtimePlot:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iterations = []
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.ion()  # インタラクティブモードをオン

    def update(self, i, train_loss, train_acc, valid_loss, valid_acc):
        self.iterations.append(i)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accs.append(valid_acc)

        self._plot()
        plt.pause(0.00001)

    def _plot(self):
        self.ax1.clear()
        self.ax1.plot(self.iterations, self.train_losses, color='C0', linestyle='-', label='Train Loss')
        self.ax1.plot(self.iterations, self.valid_losses, color='C1', linestyle='--', label='Valid Loss')
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.set_xlim(0, self.max_iter)

        self.ax2.clear()
        self.ax2.plot(self.iterations, self.train_accs, color='C0', linestyle='-', label='Train Acc')
        self.ax2.plot(self.iterations, self.valid_accs, color='C1', linestyle='--', label='Valid Acc')
        self.ax2.set_xlabel('Iterations')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.set_xlim(0, self.max_iter)
        self.ax2.set_ylim(0, 1)


def train_model(
        model,
        x_train: np.ndarray,
        t_train: np.ndarray,
        x_valid: np.ndarray,
        t_valid: np.ndarray,
        iters_num: int = 5000,
        batch_size: int = 100,
        plot_interval: int = 100,
        verbose: bool = True,
        plot: bool = True,
        name: str ="MNIST"
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    if verbose:
        print(f"Testing {name}...")

    # 結果の記録リスト
    iterations = []
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    train_size = x_train.shape[0]

    if plot:
        plotter = RealtimePlot(iters_num)

    # 学習開始
    for i in range(iters_num):

        # ミニバッチ生成
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        model.backward(x_batch, t_batch)

        # 重みパラメーター更新
        model.update_params()

        # 損失関数の値算出
        train_loss = model.loss(x_batch, t_batch)

        if i % plot_interval == 0:
            train_acc = model.accuracy(x_train, t_train)
            valid_acc = model.accuracy(x_valid, t_valid)
            valid_loss = model.loss(x_valid, t_valid)

            iterations.append(i)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

        if verbose:
            print(f'Iter:{i:>4}, '
                  f'Train [Loss:{train_loss:.4f}, Acc:{train_acc:.4f}], '
                  f'Valid [Loss:{valid_loss:.4f}, Acc:{valid_acc:.4f}]')
            if plot:
                plotter.update(i, train_loss, train_acc, valid_loss, valid_acc)

    if plot:
        plt.ioff()
        plt.show()

    return iterations, train_losses, train_accs, valid_losses, valid_accs

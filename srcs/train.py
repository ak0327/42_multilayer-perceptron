import os
import sys
sys.path.append(os.pardir)

import argparse
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from srcs.dataloader import train_test_split
from srcs.functions import Softmax
from srcs.activation import ReLU, Sigmoid
from srcs.loss import CrossEntropyLoss
from srcs.init import he_normal, xavier_normal, normal
from srcs.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from srcs.layer import Dense
from srcs.model import Sequential


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
        X_train: np.ndarray,
        t_train: np.ndarray,
        X_valid: np.ndarray,
        t_valid: np.ndarray,
        iters_num: int = 5000,
        batch_size: int = 100,
        plot_interval: int = 100,
        verbose: bool = True,
        plot: bool = True,
        name: str ="MNIST"
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    if X_train.shape[0] < batch_size:
        raise ValueError(f"Batch size {batch_size} is "
                         f"larger than the number of training samples "
                         f"({X_train.shape[0]})")

    if verbose:
        print(f"Testing {name}...")

    # 結果の記録リスト
    iterations = []
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    train_size = X_train.shape[0]

    if plot:
        plotter = RealtimePlot(iters_num)

    # 学習開始
    for epoch in range(iters_num):

        # ミニバッチ生成
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        X_batch = X_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        model.backward(X_batch, t_batch)

        # 重みパラメーター更新
        model.update_params()

        # 損失関数の値算出
        train_loss = model.loss(X_batch, t_batch)

        if epoch % plot_interval == 0:
            train_acc = model.accuracy(X_train, t_train)
            valid_acc = model.accuracy(X_valid, t_valid)
            valid_loss = model.loss(X_valid, t_valid)

            iterations.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if verbose:
                print(f'Epoch:{epoch:>4}/{iters_num}, '
                      f'Train [Loss:{train_loss:.4f}, Acc:{train_acc:.4f}], '
                      f'Valid [Loss:{valid_loss:.4f}, Acc:{valid_acc:.4f}]')
            if plot:
                plotter.update(epoch, train_loss, train_acc, valid_loss, valid_acc)

    if plot:
        plt.ioff()
        plt.show()

    return iterations, train_losses, train_accs, valid_losses, valid_accs


def _load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with np.load(file_path) as data:
            if 'X' not in data or 'y' not in data:
                raise ValueError("The data file does not contain "
                                 "'X' and 'y' arrays")

            X = data['X']
            y = data['y']

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch in number of samples: "
                                 f"X has {X.shape[0]}, "
                                 f"y has {y.shape[0]}")

            if X.shape[0] == 0:
                raise ValueError("The data is empty")

            return X, y
    except Exception as e:
        print(f"Error loading data: {str(e)}", file=sys.stderr)
        raise


def _create_model(hidden_features: list[int], learning_rate: float):
    _features = [30] + hidden_features + [2]
    _last_layer_idx = len(_features) - 2

    _layers = []
    for i in range(len(_features) - 1):
        _in_features = _features[i]
        _out_features = _features[i + 1]
        if i < _last_layer_idx:
            _activation = ReLU
            _init_method = he_normal
        else:
            _activation = Softmax
            _init_method = xavier_normal
        _layers.append(
            Dense(
                in_features=_in_features,
                out_features=_out_features,
                activation=_activation,
                init_method=_init_method
            )
        )

    _optimizer = Adam(lr=learning_rate)
    _model = Sequential(
        layers=_layers,
        criteria=CrossEntropyLoss,
        optimizer=_optimizer,
    )
    return _model


def main(
        hidden_features: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float
):
    try:
        X_train, y_train = _load_data("data/train.npz")
        X_valid, y_valid = _load_data("data/test.npz")

        model = _create_model(hidden_features, learning_rate)
        print(model.info)
        iters, train_losses, train_accs, valid_losses, valid_accs = train_model(
            model=model,
            X_train=X_train,
            t_train=y_train,
            X_valid=X_valid,
            t_valid=y_valid,
            iters_num=epochs,
            batch_size=batch_size,
            plot_interval=epochs / 100,
            verbose=True,
            plot=True,
            name="WDBC"
        )

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}", file=sys.stderr)

        print("traceback")
        _tb = e.__traceback__
        while _tb is not None:
            _filename = _tb.tb_frame.f_code.co_filename
            _line_number = _tb.tb_lineno
            print(f"File '{_filename}', line {_line_number}")
            _tb = _tb.tb_next
        print(f"Error: {str(e)}")
        sys.exit(1)


def _int_range(min_val, max_val):
    def __checker(arg):
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{arg} is not a valid integer")
        if value < min_val or value > max_val:
            raise argparse.ArgumentTypeError(f"{value} is not in range"
                                             f" [{min_val}, {max_val}]")
        return value
    return __checker


def _float_range(min_val, max_val):
    def __checker(arg):
        try:
            value = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{arg} is not a valid float")
        if value < min_val or value > max_val:
            raise argparse.ArgumentTypeError(f"{value} is not in range"
                                             f" [{min_val}, {max_val}]")
        return value
    return __checker


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process WDBC dataset for machine learning tasks"
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        nargs='*',
        default=[24, 24, 24],
        help="Number of neurons in each hidden layer "
             "(1 to 3 values, e.g., "
             "--hidden_features 24 or --hidden_features 24 24 24)"
    )
    parser.add_argument(
        "--epochs",
        type=_int_range(100, 10000),
        default=5000,
        help="Number of training epochs "
             "(integer in range [1, 10000], default: 5000)"
    )
    parser.add_argument(
        "--batch_size",
        type=_int_range(1, 100),
        default=100,
        help="Batch size for training (integer in range [1, 100], default: 100)"
    )
    parser.add_argument(
        "--learning_rate",
        type=_float_range(0.0001, 1.0),
        default=0.01,
        help="Learning rate for training "
             "(float in range [0.0001, 1.0], default: 0.01)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

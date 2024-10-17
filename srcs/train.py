import os
import sys
sys.path.append(os.pardir)

import argparse
import numpy as np


from srcs.dataloader import train_test_split, get_wdbc
from srcs.modules.functions import Softmax
from srcs.modules.activation import ReLU, Sigmoid
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.init import he_normal, xavier_normal, normal
from srcs.modules.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from srcs.modules.layer import Dense
from srcs.modules.model import Sequential
from srcs.modules.plot import RealtimePlot
from srcs.modules.io import save_training_result, save_model, load_npz, load_csv
from srcs.modules.parser import int_range, float_range, str2bool
from srcs.modules.metrics import get_metrics


np.random.seed(42)


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
        name: str = "MNIST"
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    if X_train.shape[0] < batch_size:
        raise ValueError(f"Batch size {batch_size} is "
                         f"larger than the number of training samples "
                         f"({X_train.shape[0]})")

    if verbose:
        print(f" Training {name}...")

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

        # 順伝播
        y_batch = model.forward(X_batch)

        # 損失関数の値算出
        train_loss = model.loss(y_batch, t_batch)

        # 逆伝播 勾配の計算
        model.backward()

        # 重みパラメーター更新
        model.update_params()

        if epoch % plot_interval == 0:
            y_train = model.forward(X_train)
            train_loss = model.loss(y_train, t_train)
            train_acc = model.accuracy(y_train, t_train)

            y_valid = model.forward(X_valid)
            valid_loss = model.loss(y_valid, t_valid)
            valid_acc = model.accuracy(y_valid, t_valid)

            iterations.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if verbose:
                print(f' Epoch:{epoch:>4}/{iters_num}, '
                      f'Train [Loss:{train_loss:.4f}, Acc:{train_acc:.4f}], '
                      f'Valid [Loss:{valid_loss:.4f}, Acc:{valid_acc:.4f}]')
            if plot:
                plotter.update(
                    epoch, train_loss, train_acc, valid_loss, valid_acc
                )

    if plot:
        plotter.plot()

    y_train = model.forward(X_train)
    train_acc, train_prec, train_recall, train_f1 = get_metrics(y_train, t_train)
    y_valid = model.forward(X_valid)
    valid_acc, valid_prec, valid_recall, valid_f1 = get_metrics(y_valid, t_valid)

    print(f"\nMetrics: \n"
          f" Train [Accuracy:{train_acc:.4f}, Precision:{train_prec:.4f}, Recall:{train_recall:.4f}, F1:{train_f1:.4f}]\n"
          f" Valid [Accuracy:{valid_acc:.4f}, Precision:{valid_prec:.4f}, Recall:{valid_recall:.4f}, F1:{valid_f1:.4f}]")

    return iterations, train_losses, train_accs, valid_losses, valid_accs


def _create_model(
        hidden_features: list[int],
        learning_rate: float,
        weight_decay: float,
):
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
        weight_decay=weight_decay,
    )
    return _model


def _get_train_data(
        dataset_csv_path: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_size = 0.8
    shuffle = False
    random_state = 42

    if dataset_csv_path is None:
        X, y = load_npz("data/data_train.npz")
        # X_valid, y_valid = load_npz("data/data_test.npz")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X=X,
            y=y,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_valid, y_train, y_valid = get_wdbc(
            csv_path=dataset_csv_path,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
        )
    return X_train, X_valid, y_train, y_valid


def main(
        dataset_csv_path: str | None,
        hidden_features: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        verbose: bool,
        plot: bool
):
    print(f"\n[Training]")
    try:
        X_train, X_valid, y_train, y_valid = _get_train_data(dataset_csv_path)
        model = _create_model(
            hidden_features=hidden_features,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        print(f"\n{model.info}")
        iters, train_losses, train_accs, valid_losses, valid_accs = train_model(
            model=model,
            X_train=X_train,
            t_train=y_train,
            X_valid=X_valid,
            t_valid=y_valid,
            iters_num=epochs,
            batch_size=batch_size,
            plot_interval=epochs / 100,
            verbose=verbose,
            plot=plot,
            name="WDBC"
        )

        save_training_result(
            model=model,
            iterations=iters,
            train_losses=train_losses,
            train_accs=train_accs,
            valid_losses=valid_losses,
            valid_accs=valid_accs
        )
        return model, iters, train_losses, train_accs, valid_losses, valid_accs

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process WDBC dataset for machine learning tasks"
    )
    parser.add_argument(
        "--dataset_csv_path",
        type=str,
        default=None,
        help="train csv path, if omitted load train.npz"
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
        type=int_range(100, 10000),
        default=5000,
        help="Number of training epochs "
             "(integer in range [1, 10000], default: 5000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int_range(1, 200),
        default=100,
        help="Batch size for training (integer in range [1, 200], default: 100)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float_range(0.0001, 1.0),
        default=0.01,
        help="Learning rate for training "
             "(float in range [0.0001, 1.0], default: 0.01)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float_range(0.0, 1.0),
        default=0.0,
        help="Weight decay (float in range [0.0, 1.0], default: 0.0)"
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="verbose (true/false, t/f)"
    )
    parser.add_argument(
        "--plot",
        type=str2bool,
        default=True,
        help="plot (true/false, t/f)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        dataset_csv_path=args.dataset_csv_path,
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        verbose=args.verbose,
        plot=args.plot
    )

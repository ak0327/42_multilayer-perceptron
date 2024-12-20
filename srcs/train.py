import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np


from srcs.dataloader import train_test_split, get_wdbc
from srcs.modules.functions import Softmax
from srcs.modules.activation import ReLU, Sigmoid
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.init import he_normal, xavier_normal, normal
from srcs.modules.optimizer import Optimizer, SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from srcs.modules.layer import Dense
from srcs.modules.model import Sequential
from srcs.modules.plot import RealtimePlot
from srcs.modules.io import save_training_result, load_npz, get_ndarray
from srcs.modules.parser import (
    int_range,
    float_range,
    str2bool,
    str_expected,
    validate_extention,
    int_list_type,
    valid_dir,
)
from srcs.modules.metrics import get_metrics, accuracy_score
from srcs.modules.tools import EarlyStopping


def train_model(
        model,
        X_train: np.ndarray,
        t_train: np.ndarray,
        X_valid: np.ndarray,
        t_valid: np.ndarray,
        iters_num: int = 5000,
        metrics_interval: int = 1,
        verbose: bool = True,
        plot: bool = True,
        patience: int | None = None,
        save_dir: str = "data",
        name: str = "MNIST"
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    if verbose:
        print(f" Training {name}...")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_valid shape: {X_valid.shape}\n")

    iterations = []
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    train_size = X_train.shape[0]

    if plot:
        plotter = RealtimePlot(iters_num, save_dir)

    if patience is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=False)

    # 学習開始
    for epoch in range(iters_num):
        # 順伝播
        y_train = model.forward(X_train)

        # 損失関数の値算出
        train_loss = model.loss(y_train, t_train)

        # 逆伝播 勾配の計算
        model.backward()

        # 重みパラメーター更新
        model.update_params()

        # メトリクスを計算・格納
        train_acc = accuracy_score(y_true=t_train, y_pred=y_train)

        y_valid = model.forward(X_valid)
        valid_loss = model.loss(y_valid, t_valid)
        valid_acc = accuracy_score(y_true=t_valid, y_pred=y_valid)

        iterations.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        if epoch % metrics_interval == 0 or epoch + 1 == iters_num:
            if verbose:
                print(f'  Epoch:{epoch:>4}/{iters_num}, '
                      f'Train [Loss:{train_loss:.4f}, Acc:{train_acc:.4f}], '
                      f'Valid [Loss:{valid_loss:.4f}, Acc:{valid_acc:.4f}]')
            if plot:
                plotter.update(
                    epoch, train_loss, train_acc, valid_loss, valid_acc
                )

        if patience is not None:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                model = early_stopping.best_model
                break

    if plot:
        plotter.plot()

    y_train = model.forward(X_train)
    train_acc, train_prec, train_recall, train_f1 = get_metrics(y_train, t_train)
    y_valid = model.forward(X_valid)
    valid_acc, valid_prec, valid_recall, valid_f1 = get_metrics(y_valid, t_valid)

    if verbose:
        print(f"\n"
              f" Metrics: \n"
              f"  Train - loss:{train_losses[-1]:.4f} [Accuracy:{train_acc:.4f}, Precision:{train_prec:.4f}, Recall:{train_recall:.4f}, F1:{train_f1:.4f}]\n"
              f"  Valid - loss:{valid_losses[-1]:.4f} [Accuracy:{valid_acc:.4f}, Precision:{valid_prec:.4f}, Recall:{valid_recall:.4f}, F1:{valid_f1:.4f}]\n")

    return iterations, train_losses, train_accs, valid_losses, valid_accs


def create_model(
        features: list[int],
        learning_rate: float,
        weight_decay: float = 0.0,
        optimp_str: str = "SGD",
        seed: float = 42,
        hidden_init_method: callable = normal,
        init_std: float | None = None,
):
    if len(features) < 3:
        raise ValueError(f"features must larger than 3")
    _last_layer_idx = len(features) - 2

    _layers = []
    for i in range(len(features) - 1):
        _in_features = features[i]
        _out_features = features[i + 1]
        if i < _last_layer_idx:
            _activation = ReLU
            _init_method = hidden_init_method
        else:
            _activation = Softmax
            _init_method = xavier_normal
        _layers.append(
            Dense(
                in_features=_in_features,
                out_features=_out_features,
                activation=_activation,
                init_method=_init_method,
                init_std=init_std,
                seed=seed,
            )
        )

    optimizers = {
        "SGD"       : SGD,
        "MOMENTUM"  : Momentum,
        "NESTEROV"  : Nesterov,
        "ADAGRAD"   : AdaGrad,
        "RMSPROP"   : RMSProp,
        "ADAM"      : Adam,
    }
    _optimizer = optimizers[optimp_str](lr=learning_rate)
    _model = Sequential(
        layers=_layers,
        criteria=CrossEntropyLoss,
        optimizer=_optimizer,
        weight_decay=weight_decay,
    )
    return _model


def _get_train_data(
        dataset_path: str | None,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = get_ndarray(
        wdbc_csv_path=dataset_path,
        y_onehot=True,
        drop_id=True,
        apply_normalize=True,
    )

    test_size = 0.3
    shuffle = False
    random_state = 42
    X_train, X_valid, y_train, y_valid = train_test_split(
        X=X,
        y=y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=True,
    )
    return X_train, X_valid, y_train, y_valid


def main(
        dataset_path: str | None,
        hidden_features: list[int],
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        optimp_str: str,
        verbose: bool,
        plot: bool,
        metrics_interval: int,
        patience: int | None,
        save: bool = True,
        save_dir: str = "data",
):
    print(f"\n[Train]")
    try:
        print(f" Dataset: {dataset_path}")
        X_train, X_valid, y_train, y_valid = _get_train_data(dataset_path)

        WDBC_INPUT = 30
        WDBC_OUTPUT = 2
        features = [WDBC_INPUT] + hidden_features + [WDBC_OUTPUT]
        hidden_init = normal
        init_std = 0.01

        model = create_model(
            features=features,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimp_str=optimp_str,
            seed=None,
            hidden_init_method=hidden_init,
            init_std=init_std,
        )
        if verbose:
            print(f"\n{model.info}")

        iters, train_losses, train_accs, valid_losses, valid_accs = train_model(
            model=model,
            X_train=X_train,
            t_train=y_train,
            X_valid=X_valid,
            t_valid=y_valid,
            iters_num=epochs,
            metrics_interval=metrics_interval,
            verbose=verbose,
            plot=plot,
            patience=patience,
            save_dir=save_dir,
            name="WDBC"
        )

        if save:
            save_training_result(
                model=model,
                iterations=iters,
                train_losses=train_losses,
                train_accs=train_accs,
                valid_losses=valid_losses,
                valid_accs=valid_accs,
                save_dir=save_dir
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
        description="Train on WDBC dataset with MLP"
    )
    parser.add_argument(
        "--dataset_path",
        type=validate_extention(["csv"]),
        required=True,
        help="Path to the train WBDC CSV dataset."
    )
    parser.add_argument(
        "--hidden_features",
        type=int_list_type(min_elements=2, max_elements=5, min_value=2, max_value=500),
        default=[24, 24],
        help="Number of neurons in each hidden layer in range [2, 500]"
             "(2 to 5 values, e.g., "
             "--hidden_features 24 42 or --hidden_features 24 42 24 42 24)"
    )
    parser.add_argument(
        "--epochs",
        type=int_range(100, 100000),
        default=5000,
        help="Number of training epochs "
             "(integer in range [1, 100000], default: 5000)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float_range(1e-6, 1.0),
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
        "--optimizer",
        type=str_expected(["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam"]),
        default="SGD",
        help="Optimizer ([SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam], default: SGD)"
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
    parser.add_argument(
        "--metrics_interval",
        type=int_range(1, 1000),
        default=1,
        help="metrics interval in range[1, 1000]"
    )
    parser.add_argument(
        "--patience",
        type=int_range(1, 10000),
        help="Ealry stopping patience in range[1, 10000]"
    )
    parser.add_argument(
        "--save_dir",
        type=valid_dir,
        required=True,
        help="model save dir"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        dataset_path=args.dataset_path,
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimp_str=args.optimizer,
        verbose=args.verbose,
        plot=args.plot,
        metrics_interval=args.metrics_interval,
        patience=args.patience,
        save_dir=args.save_dir,
    )

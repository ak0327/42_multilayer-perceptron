import os
import sys
sys.path.append(os.pardir)


import argparse
import numpy as np
import pandas as pd
from typing import Union, overload

from srcs.io import save_to_npz, save_to_csv, load_wdbc_data, load_csv
from srcs.parser import str2bool, float_range_exclusive


def _stratified_split(
        y: pd.Series,
        test_size: float,
        stratify: pd.Series,
        shuffle: bool,
        rng
) -> tuple[np.ndarray, np.ndarray]:
    unique_classes, class_counts = np.unique(stratify, return_counts=True)
    test_indices = []
    train_indices = []

    for cls, count in zip(unique_classes, class_counts):
        class_indices = np.where(stratify == cls)[0]
        if shuffle:
            class_indices = rng.permutation(class_indices)

        n_class_test = int(count * test_size)
        test_class_indices = class_indices[:n_class_test]
        train_class_indices = class_indices[n_class_test:]

        test_indices.extend(test_class_indices)
        train_indices.extend(train_class_indices)

    return np.array(train_indices), np.array(test_indices)


def _random_split(
        n_total: int,
        n_test: int,
        shuffle: bool,
        rng
) -> tuple[np.ndarray, np.ndarray]:
    if shuffle:
        indices = rng.permutation(n_total)
    else:
        indices = np.arange(n_total)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return train_indices, test_indices


@overload
def train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: pd.Series = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ...

@overload
def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...

def train_test_split(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        train_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: Union[pd.Series, np.ndarray] = None
) -> Union[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
           tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not (0.0 < train_size and train_size < 1.0):
        raise ValueError(
            f"ValueError: "
            f"The train_size={train_size}, should be 0.0 < train_size < 1.0")
    test_size = 1.0 - train_size

    n_total = len(X)
    n_test = int(n_total * test_size)
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        train_indices, test_indices = _stratified_split(y, test_size, stratify, shuffle, rng)
    else:
        train_indices, test_indices = _random_split(n_total, n_test, shuffle, rng)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        raise TypeError("X and y must be either "
                        "both pandas objects or both numpy arrays")
    return X_train, X_test, y_train, y_test


def get_wdbc(
        csv_path: str,
        train_size: float = 0.8,
        shuffle: bool = False,
        random_state: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # X, y = load_wdbc_data(csv_path=csv_path)
    X, y = load_csv(csv_path=csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X=X,
        y=y,
        train_size=train_size,
        shuffle=shuffle,
        stratify=y
    )
    # pd.DataFrame, pd.Seriesをnumpy配列に変換
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    return X_train, X_test, y_train, y_test


def main(
        csv_path: str,
        train_size: float,
        shuffle: bool,
        save_npz: bool,
        random_state: int
):
    print(f"\n[Loading]")
    X_train, X_test, y_train, y_test = get_wdbc(
        csv_path=csv_path,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state
    )
    try:
        if save_npz:
            save_to_npz(X=X_train, y=y_train, name="data_train")
            save_to_npz(X=X_test, y=y_test, name="data_test")
        else:
            save_to_csv(X=X_train, y=y_train, name="data_train")
            save_to_csv(X=X_test, y=y_test, name="data_test")

    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    return X_train, X_test, y_train, y_test


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process WDBC dataset for machine learning tasks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the WBDC CSV dataset"
    )
    parser.add_argument(
        "--train_size",
        type=float_range_exclusive(0.0, 1.0),
        default=0.8,
        help="Percentage of training division (float in (0.0, 1.0))"
    )
    parser.add_argument(
        "--shuffle",
        type=str2bool,
        default=True,
        help="Whether to shuffle the data before splitting (true/false, t/f)"
    )
    parser.add_argument(
        "--save_npz",
        type=str2bool,
        default=True,
        help="Save to train.npz and test.npz, othewise csv (true/false, t/f)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        csv_path=args.dataset,
        train_size=args.train_size,
        shuffle=args.shuffle,
        save_npz=args.save_npz,
        random_state=42
    )

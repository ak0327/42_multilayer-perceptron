import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import argparse
import numpy as np
import pandas as pd
from typing import Union, overload

from srcs.modules.io import save_to_npz, save_to_csv, load_wdbc_data, get_ndarray
from srcs.modules.parser import (
    str2bool,
    float_range_exclusive,
    validate_extention,
    valid_dir,
)


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


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size and test_size < 1.0):
        raise ValueError(f"test_size={test_size} should be 0.0 < test_size < 1.0")
    train_size = 1.0 - test_size

    print(f"type(X): {type(X)}")

    n_total = len(X)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    if n_train == 0:
        raise ValueError(f"train_size={train_size} results in an empty training set")
    if n_test == 0:
        raise ValueError(f"test_size={test_size} results in an empty test set")

    rng = np.random.RandomState(random_state)

    if stratify:
        train_indices, test_indices = _stratified_split(y, test_size, y, shuffle, rng)
    else:
        train_indices, test_indices = _random_split(n_total, n_test, shuffle, rng)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def count_y(y, name):
    train_b = np.sum(y == 'B')
    train_m = np.sum(y == 'M')
    print(f"\n{name} set:")
    print(f"  B: {train_b} ({train_b/len(y)*100:.1f}%)")
    print(f"  M: {train_m} ({train_m/len(y)*100:.1f}%)")


def get_wdbc(
        csv_path: str,
        test_size: float = 0.2,
        shuffle: bool = False,
        apply_normalize: bool = True,
        random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = get_ndarray(
        wdbc_csv_path=csv_path,
        y_onehot=False,
        drop_id=False,
        apply_normalize=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X=X,
        y=y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=True,
    )
    count_y(y_train, "train")
    count_y(y_test, "test")
    return X_train, X_test, y_train, y_test


def main(
        csv_path: str,
        test_size: float,
        shuffle: bool,
        save_dir: str,
        random_state: int
):
    print(f"\n[Dataloader]")
    try:
        print(f" Dataset: {csv_path}")
        print(f"  Splitting...")
        X_train, X_test, y_train, y_test = get_wdbc(
            csv_path=csv_path,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )
        print(f"  Split dataset SUCCESS\n")
        save_to_csv(X=X_train, y=y_train, dir=save_dir, name="data_train")
        save_to_csv(X=X_test, y=y_test, dir=save_dir, name="data_test")
        return X_train, X_test, y_train, y_test

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
        description="Split WDBC dataset into train and test sets"
    )
    parser.add_argument(
        "--dataset_path",
        type=validate_extention(["csv"]),
        required=True,
        help="Path to the WBDC CSV dataset"
    )
    parser.add_argument(
        "--test_size",
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
        "--save_dir",
        type=valid_dir,
        required=True,
        help="dataset save dir"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        csv_path=args.dataset_path,
        test_size=args.test_size,
        shuffle=args.shuffle,
        save_dir=args.save_dir,
        random_state=42
    )

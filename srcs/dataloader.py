import os
import sys
sys.path.append(os.pardir)


import argparse
import numpy as np
import pandas as pd
from itertools import product


def _load_wdbc_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    expected_shape = (569, 32)
    if df.shape != expected_shape:
        raise ValueError(f"Invalid data shape. "
                         f"Expected {expected_shape}, but got {df.shape}")

    columns = ['id', 'diagnosis']
    # 特徴量の名前リスト
    features = [
        'radius',
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concavePoints',
        'symmetry',
        'fractal_dimension'
    ]
    # 統計量の名前リスト
    stats = ['mean', 'stderr', 'worst']

    for feature, stat in product(features, stats):
        columns.append(f"{feature}_{stat}")

    df.columns = columns
    df = df.drop('id', axis=1)

    # ラベルを数値に変換（M -> 1, B -> 0）
    y = pd.to_numeric(df['diagnosis'].map({'M': 1, 'B': 0}), downcast='integer')

    X = df.drop('diagnosis', axis=1)
    return X, y


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


def _train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: pd.Series = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test  = y.iloc[train_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test


def get_wdbc(
        csv_path: str,
        train_size: float = 0.8,
        shuffle: bool = False,
        random_state: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = _load_wdbc_data(csv_path=csv_path)
    X_train, X_test, y_train, y_test = _train_test_split(
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


def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _float_0_to_1(s):
    try:
        float_num = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s} not a floating-point literal")

    if float_num <= 0.0 or 1.0 <= float_num:
        raise argparse.ArgumentTypeError(f"{float_num} not in range (0.0, 1.0)")
    return float_num


def _save_to_npz(X: np.ndarray, y: np.ndarray, name: str):
    try:
        np.savez(f"{name}.npz", X=X, y=y)
        print(f"Train data saved to {os.path.abspath(f'{name}.npz')}")
    except IOError as e:
        raise IOError(f"fail to saving {name} data: {e}")


def main(csv_path: str, train_size: float, shuffle: bool, random_state: int):
    X_train, X_test, y_train, y_test = get_wdbc(
        csv_path=csv_path,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state
    )
    try:
        _save_to_npz(X=X_train, y=y_train, name="train")
        _save_to_npz(X=X_test, y=y_test, name="test")
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
        type=_float_0_to_1,
        default=0.8,
        help="Percentage of training division (float in (0.0, 1.0))"
    )
    parser.add_argument(
        "--shuffle",
        type=_str2bool,
        default=True,
        help="Whether to shuffle the data before splitting (true/false, t/f)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        csv_path=args.dataset,
        train_size=args.train_size,
        shuffle=args.shuffle,
        random_state=42
    )

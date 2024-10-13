import os
import sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
from itertools import product


print(os.getcwd())


def load_wdbc_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, "data", "data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

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

    y = df['diagnosis']
    y = y.replace({'M': 1, 'B': 0}).astype(int)

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


def train_test_split(
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
            f"The train_size={train_size}, should be 0.0 < valid_size < 1.0")
    valid_size = 1.0 - train_size

    n_total = len(X)
    n_valid = int(n_total * valid_size)
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        train_indices, valid_indices = _stratified_split(y, valid_size, stratify, shuffle, rng)
    else:
        train_indices, valid_indices = _random_split(n_total, n_valid, shuffle, rng)

    X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
    y_train, y_valid  = y.iloc[train_indices], y.iloc[valid_indices]
    return X_train, X_valid, y_train, y_valid


def get_wdbc(
        train_size: float = 0.8,
        shuffle: bool = False,
        random_state: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_wdbc_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X=X,
        y=y,
        train_size=train_size,
        shuffle=shuffle,
        stratify=y
    )
    # pd.DataFrame, pd.Seriesをnumpy配列に変換
    X_train = X_train.values
    X_valid = X_valid.values
    y_train = y_train.values
    y_valid = y_valid.values

    return X_train, X_valid, y_train, y_valid

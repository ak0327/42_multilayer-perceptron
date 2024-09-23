import numpy as np
import pandas as pd
from itertools import product


def get_wdbc_df(csv_path: str) -> tuple[pd.DataFrame: pd.Series]:
    try:
        _df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        raise RuntimeError(f"Error occurred while reading csv file: {str(e)}")

    expected_shape = (568, 32)
    if _df.shape != expected_shape:
        raise ValueError(f"Unexpected csv data: {_df.shape},"
                         f" Expected {expected_shape}")

    _columns = ['id', 'diagnosis']
    _features = [
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
    _stats = ['mean', 'stderr', 'worst']

    for feature, stat in product(_features, _stats):
        _columns.append(f"{feature}_{stat}")

    df.columns = columns
    _y = df['diagnosis']
    _X = df.drop(columns=['diagnosis'])
    return _X, _y


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
        test_size: float,
        shuffle: bool = False,
        random_state: int = None,
        stratify: pd.Series = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not (0.0 < test_size and test_size < 1.0):
        raise ValueError(
            f"ValueError: "
            f"The test_size={test_size}, should be 0.0 < test_size < 1.0")

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

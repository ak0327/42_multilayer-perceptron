import os
import sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import pickle
from itertools import product
from typing import Union

from srcs.modules.model import Sequential
from srcs.modules.tools import normalize


def save_to_npz(X: np.ndarray, y: np.ndarray, dir: str, name: str):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples, but got {X.shape[0]} and {y.shape[0]}.")

    try:
        path = f"{dir}/{name}.npz"
        np.savez(path, X=X, y=y)
        print(f" {name.capitalize()} data saved to {os.path.abspath(path)}")
    except IOError as e:
        raise IOError(f"fail to saving {name} data: {e}")


def load_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")

        with np.load(npz_path) as data:
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

            print(f"Load successd: {npz_path}")
            return X, y
    except Exception as e:
        print(f"Error loading data: {str(e)}", file=sys.stderr)
        raise


def save_to_csv(X: np.ndarray, y: np.ndarray, dir: str, name: str):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples, but got {X.shape[0]} and {y.shape[0]}.")

    try:
        df = pd.DataFrame(X)
        df['diagnosis'] = y

        path = f"{dir}/{name}.csv"
        df.to_csv(path, index=False, header=False)
        print(f" {name.capitalize()} data saved to {os.path.abspath(path)}")
    except IOError as e:
        raise IOError(f"Failed to save {name} data: {str(e)}")


def get_ndarray(
        wdbc_csv_path: str,
        y_onehot: bool = True,
        drop_id: bool = True,
        apply_normalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_wdbc_data(
        wdbc_csv_path,
        y_onehot=y_onehot,
        drop_id=drop_id,
        apply_normalize=apply_normalize
    )

    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def normalize_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    columns = ['id', 'diagnosis']
    # 特徴量
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
    # 統計量
    stats = ['mean', 'stderr', 'worst']

    for feature, stat in product(features, stats):
        columns.append(f"{feature}_{stat}")

    df.columns = columns

    feature_columns = [col for col in columns if col not in ['id', 'diagnosis']]
    df = normalize(df, feature_columns)

    output_path = csv_path.rsplit('.', 1)[0] + '_normalized.csv'
    df.to_csv(output_path, index=False, header=False, float_format='%.10f')


def split_csv(csv_path, train_size: float, shuffle: bool, stratify: bool):
    X, y = load_wdbc_data(csv_path=csv_path, y_onehot=False, drop_id=False)

    _stratify = None
    if stratify:
        _stratify = y

    X_train, X_test, y_train, y_test = dataloader.train_test_split(
        X=X,
        y=y,
        train_size=train_size,
        shuffle=shuffle,
        stratify=_stratify,
    )
    save_to_csv(X=X_train, y=y_train, dir="../data", name="data_raw_train")
    save_to_csv(X=X_test, y=y_test, dir="../data", name="data_raw_test")


def load_wdbc_data(
        csv_path: str,
        y_onehot: bool,
        drop_id: bool,
        apply_normalize: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    diagnosis_idx = None
    for i, col in enumerate(df.columns):
        if df[col].isin(["M", "B"]).any():
            diagnosis_idx = i
            break
    if diagnosis_idx is None:
        raise ValueError("'diagnosis' not found")

    y = df[diagnosis_idx].rename('diagnosis')
    if y_onehot:
        y = pd.to_numeric(y.map({'M': 1, 'B': 0}), downcast='integer')

    X = df.drop(diagnosis_idx, axis=1)

    columns = ['id']
    # 特徴量
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
    # 統計量
    stats = ['mean', 'stderr', 'worst']

    for feature, stat in product(features, stats):
        columns.append(f"{feature}_{stat}")

    X.columns = columns

    if apply_normalize:
        feature_columns = [col for col in columns if col not in ['id']]
        X = normalize(X, feature_columns)

    if drop_id:
        X = X.drop('id', axis=1)

    return X.astype(np.float64), y


def save_training_result(
        model,
        iterations: list[int],
        train_losses: list[float],
        train_accs: list[float],
        valid_losses: list[float],
        valid_accs: list[float],
        save_dir: str,
):
    model_save_path = f"{save_dir}/model.pkl"
    save_model(model, model_save_path)
    print(f" Model data saved to {os.path.abspath(model_save_path)}")

    metrics_save_path = f"{save_dir}/metrics.npz"
    np.savez(
        metrics_save_path,
        iterations=iterations,
        train_losses=train_losses,
        train_accs=train_accs,
        valid_losses=valid_losses,
        valid_accs=valid_accs
    )
    print(f" Metrics saved to {os.path.abspath(metrics_save_path)}")


def load_training_result(save_dir):
    data = np.load(f"{save_dir}/training_results.npz")
    iterations = data['iterations']
    train_losses = data['train_losses']
    train_accs = data['train_accs']
    valid_losses = data['valid_losses']
    valid_accs = data['valid_accs']


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

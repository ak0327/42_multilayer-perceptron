import os
import sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import pickle
from itertools import product
from typing import Union

from srcs.model import Sequential


TRAIN_RESULT_PATH = "data/training_results.npz"
MODEL_PATH = "data/model.pkl"


def save_to_npz(X: np.ndarray, y: np.ndarray, name: str):
    try:
        path = f"data/{name}.npz"
        np.savez(path, X=X, y=y)
        print(f"{name.capitalize()} data saved to {os.path.abspath(path)}")
    except IOError as e:
        raise IOError(f"fail to saving {name} data: {e}")


def load_npz(npz_path: str) -> tuple[np.ndarray, : np.ndarray]:
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


def save_to_csv(X: np.ndarray, y: np.ndarray, name: str):
    try:
        df = pd.DataFrame(X)
        df['diagnosis'] = y

        path = f"data/{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name.capitalize()} data saved to {os.path.abspath(path)}")
    except IOError as e:
        raise IOError(f"Failed to save {name} data: {e}")


def load_csv(
        csv_path: str,
        np: bool = False
) -> Union[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
           tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        df = pd.read_csv(csv_path, index_col=None)
        print(f"Load successd: {csv_path}")
        if 'id' in df.columns or df.shape[1] == 32:
            # 元のデータ形式の場合
            X, y = load_wdbc_data(csv_path)
        else:
            # 保存したデータの場合
            y = df['diagnosis'].values
            X = df.drop('diagnosis', axis=1).values

        if np and isinstance(X, pd.DataFrame):
            X = X.values
        if np and isinstance(y, pd.DataFrame):
            y = y.values
        return X, y
    except ValueError:
        raise IOError(f"Failed to load {csv_path}: {e}")


def load_wdbc_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    # expected_shape = (569, 32)
    # if df.shape[1] != expected_shape[1]:
    #     raise ValueError(f"Invalid data shape. "
    #                      f"Expected {expected_shape}, but got {df.shape}")

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


def save_training_result(
        model,
        iterations: list[int],
        train_losses: list[float],
        train_accs: list[float],
        valid_losses: list[float],
        valid_accs: list[float]
):
    save_model(model, MODEL_PATH)
    np.savez(
        TRAIN_RESULT_PATH,
        iterations=iterations,
        train_losses=train_losses,
        train_accs=train_accs,
        valid_losses=valid_losses,
        valid_accs=valid_accs
    )


def load_training_result():
    data = np.load(TRAIN_RESULT_PATH)
    iterations = data['iterations']
    train_losses = data['train_losses']
    train_accs = data['train_accs']
    valid_losses = data['valid_losses']
    valid_accs = data['valid_accs']


def save_model(model, filename=MODEL_PATH):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename=MODEL_PATH):
    with open(filename, 'rb') as f:
        return pickle.load(f)

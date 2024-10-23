import numpy as np
import pandas as pd
import pytest

from srcs.dataloader import get_wdbc
from srcs.modules.io import load_wdbc_data, save_to_npz, save_to_csv


def _evaluate_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        test_size: float,
        shuffle: bool = False,
        random_state: int = None
):
    total_samples = len(X_train) + len(X_valid)
    actual_test_size = len(X_valid) / total_samples

    print(f"total size   : {total_samples}")
    print(f"X_train size : {len(X_train)}")
    print(f"X_valid size : {len(X_valid)}")
    print(f"y_train size : {len(y_train)}")
    print(f"y_valid size : {len(y_valid)}")
    print(f"train size:\n"
          f" expected: {test_size:.2f}\n"
          f" actual  : {actual_test_size:.2f}\n"
          f" error   : {abs(actual_test_size - test_size):.4f}\n")

    # サイズの一致を確認
    assert len(X_train) == len(y_train), "Mismatch in sizes of X_train and y_train"
    assert len(X_valid) == len(y_valid), "Mismatch in sizes of X_valid and y_valid"
    assert abs(actual_test_size - test_size) < 0.01, "Test size deviates significantly from expected value"

    # shuffleのチェック
    if not shuffle:
        assert (X_train == X.iloc[:len(X_train)].values).all(), "X_train is not in original order when shuffle=False"
        assert (X_valid == X.iloc[len(X_train):].values).all(), "X_valid is not in original order when shuffle=False"
        print("Shuffle=False: Order is preserved correctly.")
    else:
        assert not (X_train == X.iloc[:len(X_train)].values).all(), "X_train is in original order when shuffle=True"
        assert not (X_valid == X.iloc[len(X_train):].values).all(), "X_valid is in original order when shuffle=True"
        print("Shuffle=True: Data has been shuffled correctly.")

    # stratifyのチェック (クラス分布の確認)
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    valid_dist = pd.Series(y_valid).value_counts(normalize=True)
    full_dist = y.value_counts(normalize=True)

    for cls in full_dist.index:
        assert abs(train_dist.get(cls, 0) - full_dist[cls]) < 0.05, f"Stratification issue in train for class {cls}"
        assert abs(valid_dist.get(cls, 0) - full_dist[cls]) < 0.05, f"Stratification issue in test for class {cls}"
    print("Stratification: Class distributions are correctly maintained in train and test splits.")

    print("All checks passed. Split performed correctly.\n\n")

    # # 分布の表示
    # print(f"y_train distribution: {y_train.value_counts()}\n")
    # print(f"y_valid distribution : {y_valid.value_counts()}\n")
    #
    # train_percentages = y_train.value_counts(normalize=True) * 100
    # test_percentages = y_valid.value_counts(normalize=True) * 100
    # print(f"Percentage distribution in y_train: {train_percentages}\n")
    # print(f"Percentage distribution in y_valid: {test_percentages}\n")


def test_dataloader_shuffle():
    X, y = load_wdbc_data(
        csv_path="data/data.csv",
        y_onehot=False,
        drop_id=False,
        apply_normalize=False,
    )

    random_state = 42
    for test_size in [0.1, 0.8, 0.9]:
        X_train, X_valid, y_train, y_valid = get_wdbc(
            csv_path="data/data.csv",
            test_size=test_size,
            shuffle=True,
            random_state=random_state
        )
        _evaluate_split(
            X=X,
            y=y,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            test_size=test_size,
            shuffle=True,
            random_state=random_state
        )


def test_invalid_size():
    invalid_sizes = [-np.inf, -1.0, -0.1, 0.0, 1.0, 1.1, np.inf, np.nan]
    for test_size in invalid_sizes:
        with pytest.raises(ValueError, match=r".*should be 0.0 < test_size < 1.0"):
            _, _, _, _ = get_wdbc(
                csv_path="data/data.csv",
                test_size=test_size,
            )

    invalid_sizes = [0.001]
    for test_size in invalid_sizes:
        with pytest.raises(ValueError, match=r".*results in an empty test set"):
            _, _, _, _ = get_wdbc(
                csv_path="data/data.csv",
                test_size=test_size,
            )


def test_invalid_csv_path():
    # invalid_paths = [""]
    invalid_paths = ["data.csv", "data.csvv"]
    for path in invalid_paths:
        with pytest.raises(FileNotFoundError, match=r".*CSV file not found at path:*"):
            _, _, _, _ = get_wdbc(csv_path=path)


def test_invalid_save_dir():
    X_train, X_test, y_train, y_test = get_wdbc(
        csv_path="data/data.csv",
        test_size=0.8,
        shuffle=False,
        random_state=42
    )

    # invalid_paths = [""]
    invalid_paths = ["nothing", None]
    for path in invalid_paths:
        with pytest.raises(IOError):
            save_to_npz(X=X_train, y=y_train, dir=path, name="pytest_data")

    invalid_paths = ["data"]
    for path in invalid_paths:
        with pytest.raises(ValueError, match=r".*X and y must have the same number of samples"):
            save_to_npz(X=X_train, y=y_test, dir=path, name="pytest_data")

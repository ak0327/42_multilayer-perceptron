import numpy as np
import pandas as pd

from srcs.dataloader import get_wdbc, load_wdbc_data


def evaluate_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        train_size: float,
        shuffle: bool = False,
        random_state: int = None
):
    total_samples = len(X_train) + len(X_valid)
    actual_train_size = len(X_train) / total_samples


    print(f"total size   : {total_samples}")
    print(f"X_train size : {len(X_train)}")
    print(f"X_valid size : {len(X_valid)}")
    print(f"y_train size : {len(y_train)}")
    print(f"y_valid size : {len(y_valid)}")
    print(f"train size:\n"
          f" expected: {train_size:.2f}\n"
          f" actual  : {actual_train_size:.2f}\n"
          f" error   : {abs(actual_train_size - train_size):.4f}\n")

    # サイズの一致を確認
    assert len(X_train) == len(y_train), "Mismatch in sizes of X_train and y_train"
    assert len(X_valid) == len(y_valid), "Mismatch in sizes of X_valid and y_valid"
    assert abs(actual_train_size - train_size) < 0.01, "Test size deviates significantly from expected value"

    # shuffleのチェック
    if not shuffle:
        assert (X_train.index == X.index[:len(X_train)]).all(), "X_train is not in original order when shuffle=False"
        assert (X_valid.index == X.index[len(X_train):]).all(), "X_valid is not in original order when shuffle=False"
        print("Shuffle=False: Order is preserved correctly.")
    else:
        assert not (X_train.index == X.index[:len(X_train)]).all(), "X_train is in original order when shuffle=True"
        assert not (X_valid.index == X.index[len(X_train):]).all(), "X_valid is in original order when shuffle=True"
        print("Shuffle=True: Data has been shuffled correctly.")

    # stratifyのチェック
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_valid.value_counts(normalize=True)
    full_dist = y.value_counts(normalize=True)

    for cls in full_dist.index:
        assert abs(train_dist.get(cls, 0) - full_dist[cls]) < 0.05, f"Stratification issue in train for class {cls}"
        assert abs(test_dist.get(cls, 0) - full_dist[cls]) < 0.05, f"Stratification issue in test for class {cls}"
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
    X, y = load_wdbc_data()

    random_state = 42
    for train_size in [0.1, 0.8, 0.9]:
        X_train, X_valid, y_train, y_valid = get_wdbc(
            train_size=train_size,
            shuffle=True,
            random_state=random_state
        )
        evaluate_split(
            X=X,
            y=y,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            train_size=train_size,
            shuffle=True,
            random_state=random_state
        )

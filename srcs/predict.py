import os
import sys
sys.path.append(os.pardir)

import argparse
import numpy as np

from srcs.dataloader import train_test_split
from srcs.functions import Softmax
from srcs.activation import ReLU, Sigmoid
from srcs.loss import CrossEntropyLoss
from srcs.init import he_normal, xavier_normal, normal
from srcs.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from srcs.layer import Dense
from srcs.model import Sequential
from srcs.plot import RealtimePlot
from srcs.io import load_model, load_wdbc_data, load_npz, load_csv


def predict(
        model: Sequential,
        X_test: np.ndarray,
        t_test: np.ndarray,
        name: str ="MNIST"
):
    print(f" Predicting {name}...")
    accuracy = model.accuracy(x=X_test, t=t_test)
    print(f" Accuracy: {accuracy: .4f}")
    return accuracy


def main(
        dataset_csv_path: str | None,
        model_path: str
):
    print(f"\n[Prediction]")
    if dataset_csv_path is None:
        X_test, y_test = load_npz("data/data_test.npz")
    else:
        X_test, y_test = load_csv(dataset_csv_path, np=True)

    model: Sequential = load_model(model_path)
    accuracy = predict(
        model=model,
        X_test=X_test,
        t_test=y_test,
        name="WDBC"
    )
    return accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process WDBC dataset for machine learning tasks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset_csv_path",
        type=str,
        help="Path to the predict WBDC CSV dataset, "
             "if omitted use test.npz"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        model_path=args.model_path,
        dataset_csv_path=args.dataset_csv_path,
        dataset_npz_path=args.dataset_npz_path,
    )

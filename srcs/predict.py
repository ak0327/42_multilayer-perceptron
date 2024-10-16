import os
import sys
sys.path.append(os.pardir)

import argparse
import numpy as np

from srcs.dataloader import train_test_split
from srcs.modules.functions import Softmax
from srcs.modules.activation import ReLU, Sigmoid
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.init import he_normal, xavier_normal, normal
from srcs.modules.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from srcs.modules.layer import Dense
from srcs.modules.model import Sequential
from srcs.modules.plot import RealtimePlot
from srcs.modules.io import load_model, load_wdbc_data, load_npz, load_csv


def predict(
        model: Sequential,
        X_test: np.ndarray,
        t_test: np.ndarray,
        name: str ="MNIST"
) -> float:
    print(f" Predicting {name}...")
    accuracy = model.accuracy(x=X_test, t=t_test)
    print(f" Accuracy: {accuracy: .4f}")
    return accuracy


def main(
        dataset_csv_path: str | None,
        model_path: str
) -> float:
    print(f"\n[Prediction]")
    try:
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

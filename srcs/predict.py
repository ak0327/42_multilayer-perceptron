import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from srcs.modules.metrics import get_metrics
from srcs.modules.parser import validate_extention


def predict(
        model: Sequential,
        X_test: np.ndarray,
        t_test: np.ndarray,
        name: str ="MNIST"
) -> float:
    print(f" Predicting {name}...")
    y_pred = model.forward(X_test)

    accuracy, precision, recall, f1_score = get_metrics(y=y_pred, t=t_test)
    print(f"  Pred [Accuracy:{accuracy:.4f}, Precision:{precision:.4f}, Recall:{recall:.4f}, F1:{f1_score:.4f}]")
    return accuracy


def main(
        dataset_path: str,
        model_path: str
) -> float:
    print(f"\n[Predict]")
    try:
        print(f" Dataset: {dataset_path}\n")
        if dataset_path.endswith(".npz"):
            X_test, y_test = load_npz(dataset_path)
        elif dataset_path.endswith(".csv"):
            X_test, y_test = load_csv(dataset_path, np=True)
        else:
            raise ValueError(f"Invalid file path: Required npz or csv")

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
        description="Predict with trained model"
    )
    parser.add_argument(
        "--model_path",
        type=validate_extention(["pkl"]),
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset_path",
        type=validate_extention(["npz", "csv"]),
        required=True,
        help="Path to the predict WBDC CSV or NPZ dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
    )

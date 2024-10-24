import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs import dataloader, train, predict
import random
import numpy as np


random.seed(42)
np.random.seed(42)


def run_dataloader():
    sys.argv = [
        'dataloader.py',
        '--dataset_path',   'data/data.csv',
        '--test_size',      '0.3',
        '--shuffle',        'true',
        '--save_dir',       'data',
    ]
    args = dataloader.parse_arguments()
    _, _, _, _ = dataloader.main(
        csv_path=args.dataset_path,
        test_size=args.test_size,
        shuffle=args.shuffle,
        save_dir=args.save_dir,
        random_state=42
    )


def run_train():
    sys.argv = [
        'train.py',
        '--dataset_path',       'data/data_train.csv',
        '--hidden_features',    '10 2',
        '--epochs',             '4000',
        '--learning_rate',      '1e-4',
        '--optimizer',          'Adam',
        '--verbose',            'true',
        '--plot',               'true',
        '--metrics_interval',   '100',
        '--save_dir',           'data',
    ]
    args = train.parse_arguments()
    train.main(
        dataset_path=args.dataset_path,
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimp_str=args.optimizer,
        verbose=args.verbose,
        plot=args.plot,
        metrics_interval=args.metrics_interval,
        patience=args.patience,
        save_dir=args.save_dir,
    )


def run_predict():
    sys.argv = [
        'predict.py',
        '--model_path',     'data/model.pkl',
        '--dataset_path',   'data/data_test.csv',
    ]
    args = predict.parse_arguments()
    predict.main(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
    )


def main():
    try:
        run_dataloader()
        run_train()
        run_predict()

    except Exception as e:
        print(f"fatal error: {str(e)}")
        print("traceback")
        _tb = e.__traceback__
        while _tb is not None:
            _filename = _tb.tb_frame.f_code.co_filename
            _line_number = _tb.tb_lineno
            print(f"File '{_filename}', line {_line_number}")
            _tb = _tb.tb_next
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

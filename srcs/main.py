import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from srcs import dataloader, train, predict, evaluation
from srcs.modules.io import normalize_data


def run_dataloader():
    sys.argv = [
        'dataloader.py',
        '--dataset_path',   'data/data.csv',
        '--test_size',      '0.25',
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


def run_train(dataset_path):
    sys.argv = [
        'train.py',
        '--dataset_path',   'data/data_train.csv',
        # '--dataset_path',   'data/data_normalized_train.csv',
        # '--hidden_features',    '50 30 20 5',
        # '--hidden_features',    '50 30 10',
        # '--hidden_features',    '20 10 5',
        # '--hidden_features',    '100 15',
        # '--hidden_features',    '15 2',
        '--hidden_features',    '10 2',
        # '--hidden_features',    '50 10 5 3',
        # '--hidden_features',    '200 50',
        # '--hidden_features',    '300 200 50',
        # '--hidden_features',    '50 5',
        '--epochs',             '5000',
        '--learning_rate',      '1e-4',
        '--optimizer',          'Adam',
        # '--optimizer',          'SGD',

        # '--hidden_features',    '20 10 5',
        # '--epochs',             '5000',
        # '--learning_rate',      '1e-4',
        # '--optimizer',          'Adam',


        # '--weight_decay',       '1e-6',
        # '--hidden_features',    '20 10 5',
        # '--epochs',             '5000',
        # '--learning_rate',      '1e-3',
        # '--optimizer',          'Nesterov',

        '--verbose',            'false',
        '--plot',               'false',
        '--metrics_interval',   '100',
        '--save_dir',           'data',
        # '--patience',           '100',
        # '--patience',           '50',
    ]
    args = train.parse_arguments()
    train.main(
        dataset_path=dataset_path,
        # dataset_path=args.dataset_path,
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


def run_predict(dataset_path):
    sys.argv = [
        'predict.py',
        '--model_path',     'data/model.pkl',
        '--dataset_path',   'data/data_test.csv',
        # '--dataset_path',   'data/data_train.csv',
        # '--dataset_path',   'data/data_normalized_test.csv',
    ]
    args = predict.parse_arguments()
    predict.main(
        model_path=args.model_path,
        # dataset_path=args.dataset_path,
        dataset_path=dataset_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="main"
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="eval"
    )
    args = parser.parse_args()
    try:
        if args.eval is None:
            print("train mode")
            run_dataloader()
            train_dataset = 'data/data_train.csv'
            test_dataset = 'data/data_test.csv'

        elif args.eval == "n":
            print("eval mode normalize")
            normalize_data("data/data.csv")
            evaluation.splitDataset("data/data_normalized.csv", cut=0.25, label=False, shuffle=True)
            train_dataset = 'data/data_normalized_train.csv'
            test_dataset = 'data/data_normalized_test.csv'

        else:
            print("eval mode")
            evaluation.splitDataset("data/data.csv", cut=0.25, label=False, shuffle=True)
            train_dataset = 'data/data_train.csv'
            test_dataset = 'data/data_test.csv'

        run_train(train_dataset)
        run_predict(test_dataset)

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

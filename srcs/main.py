import sys
from srcs import dataloader, train


def run_dataloader():
    sys.argv = [
        'dataloader.py',
        '--dataset', 'data/data.csv',
        '--train_size', '0.8',
        '--shuffle', 'true'
    ]
    args = dataloader.parse_arguments()
    _, _, _, _ = dataloader.main(
        csv_path=args.dataset,
        train_size=args.train_size,
        shuffle=args.shuffle,
        random_state=42
    )


def run_train():
    sys.argv = [
        'train.py',
        '--hidden_features', '50',
        '--epochs', '5000',
        '--batch_size', '100',
        '--learning_rate', '0.001',
    ]
    args = train.parse_arguments()
    train.main(
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


def run_predict():
    # todo
    pass


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
        exit(1)


if __name__ == '__main__':
    main()

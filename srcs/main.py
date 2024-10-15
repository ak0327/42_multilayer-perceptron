import sys
import dataloader


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
    # todo
    pass


def run_predict():
    # todo
    pass


def main():
    run_dataloader()
    run_train()
    run_predict()


if __name__ == '__main__':
    main()

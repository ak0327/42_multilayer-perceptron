import sys
from srcs import dataloader, train, predict


def _run_dataloader():
    sys.argv = [
        'dataloader.py',
        '--dataset', 'data/data.csv',
        '--train_size', '0.8',
        '--shuffle', 'true',
        '--save_npz', 'false',
    ]
    args = dataloader.parse_arguments()
    _, _, _, _ = dataloader.main(
        csv_path=args.dataset,
        train_size=args.train_size,
        shuffle=args.shuffle,
        save_npz=args.save_npz,
        random_state=42
    )


def _run_train():
    sys.argv = [
        'train.py',
        '--dataset_csv_path', 'data/data_train.csv',
        '--hidden_features', '50', '30',
        '--epochs', '5000',
        '--batch_size', '100',
        '--learning_rate', '0.0001',
        '--verbose', 'false',
        '--plot', 'false',
    ]
    args = train.parse_arguments()
    train.main(
        dataset_csv_path=args.dataset_csv_path,
        hidden_features=args.hidden_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        plot=args.plot
    )


def _run_predict():
    sys.argv = [
        'predict.py',
        '--model_path', 'data/model.pkl',
        '--dataset_csv_path', 'data/data_test.csv',
    ]
    args = predict.parse_arguments()
    accuracy = predict.main(
        model_path=args.model_path,
        dataset_csv_path=args.dataset_csv_path,
    )
    return accuracy


def test_csv():
    _run_dataloader()
    _run_train()
    accuracy = _run_predict()
    assert 0.93 <= accuracy

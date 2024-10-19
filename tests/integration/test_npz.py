import sys
from srcs import dataloader, train, predict


def _run_dataloader():
    sys.argv = [
        'dataloader.py',
        '--dataset', 'data/data.csv',
        '--train_size', '0.8',
        '--shuffle', 'true',
        '--save_npz', 'true',
        '--save_dir', 'data',
    ]
    args = dataloader.parse_arguments()
    _, _, _, _ = dataloader.main(
        csv_path=args.dataset,
        train_size=args.train_size,
        shuffle=args.shuffle,
        save_npz=args.save_npz,
        save_dir=args.save_dir,
        random_state=42
    )


def _run_train():
    sys.argv = [
        'train.py',
        '--dataset_path', 'data/data_train.npz',
        '--hidden_features', '50', '30',
        '--epochs', '5000',
        '--learning_rate', '0.001',
        '--verbose', 'false',
        '--plot', 'false',
        '--metrics_interval', '100',
        '--save_dir', 'data',
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


def _run_predict():
    sys.argv = [
        'predict.py',
        '--model_path', 'data/model.pkl',
        '--dataset_path', 'data/data_test.npz',
    ]
    args = predict.parse_arguments()
    accuracy = predict.main(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
    )
    return accuracy


def test_csv():
    _run_dataloader()
    _run_train()
    accuracy = _run_predict()
    assert 0.90 <= accuracy

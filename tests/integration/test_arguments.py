import pytest
import sys
import numpy as np
from srcs import dataloader, train, predict


def dict_to_argv(file, arg_dict):
    argv = [file]
    for key, value in arg_dict.items():
        if value is None:
            continue
        argv.append(f'--{key}')
        argv.append(str(value))
    return argv


class TestDataloaderArgument:
    @classmethod
    def setup_class(cls):
        cls.base_args = {
            'dataset_path': 'data/data.csv',
            'test_size': '0.2',
            'shuffle': 'true',
            'save_dir': 'data'
        }
        cls.filename = 'dataloader.py'

    def test_dataloader_arguments(self):
        sys.argv = dict_to_argv(self.filename, self.base_args)
        args = dataloader.parse_arguments()
        assert args.dataset_path == 'data/data.csv'
        assert args.test_size == 0.2
        assert args.shuffle
        assert args.save_dir == 'data'

    @pytest.mark.parametrize("field, value, expected_error", [
        ('dataset_path',  None,         SystemExit),
        ('dataset_path', '',            SystemExit),
        ('dataset_path', 'nothing',     SystemExit),
        ('dataset_path', 'notcsv.npz',  SystemExit),

        ('test_size',  '',     SystemExit),
        ('test_size',  '0',    SystemExit),
        ('test_size',  '-0.1', SystemExit),
        ('test_size',  '1.2',  SystemExit),
        ('test_size',  'inf',  SystemExit),
        ('test_size',  'nan',  SystemExit),

        ('shuffle',     '',     SystemExit),

        ('save_dir',    None,       SystemExit),
        ('save_dir',    'nothing',  SystemExit), ])
    def test_invalid_arguments(self, field, value, expected_error):
        invalid_args = self.base_args.copy()
        invalid_args[field] = value
        sys.argv = dict_to_argv(self.filename, invalid_args)
        with pytest.raises(expected_error):
            dataloader.parse_arguments()

    @pytest.mark.parametrize("field, value, expected", [
        ('test_size', '0.5', 0.5),
        ('shuffle', 'false', False), ])
    def test_valid_argument_variations(self, field, value, expected):
        valid_args = self.base_args.copy()
        valid_args[field] = value
        sys.argv = dict_to_argv(self.filename, valid_args)
        args = dataloader.parse_arguments()
        assert getattr(args, field) == expected


class TestTrainArgument:
    @classmethod
    def setup_class(cls):
        cls.base_args = {
            'dataset_path':     'data/data_train.csv',
            'hidden_features':  '50 30',
            'epochs':           '2000',
            'learning_rate':    '0.0001',
            'weight_decay':     '0.001',
            'optimizer':        'Adam',
            'verbose':          'false',
            'plot':             'false',
            'metrics_interval': '10',
            'patience':         '100',
            'save_dir':         'data',
        }
        cls.filename = 'train.py'

    def test_train_arguments(self):
        sys.argv = dict_to_argv(self.filename, self.base_args)
        args = train.parse_arguments()
        assert args.dataset_path == 'data/data_train.csv'
        assert args.hidden_features == [50, 30]
        assert args.epochs == 2000
        assert args.learning_rate == 0.0001
        assert args.weight_decay == 0.001
        assert args.optimizer == 'ADAM'
        assert not args.verbose
        assert not args.plot
        assert args.metrics_interval == 10
        assert args.patience == 100
        assert args.save_dir == 'data'

    @pytest.mark.parametrize("field, value, expected_error", [
        ('dataset_path',    None,           SystemExit),
        ('dataset_path',    '',             SystemExit),
        ('dataset_path',    ' ',            SystemExit),
        ('dataset_path',    'nothing',      SystemExit),
        ('dataset_path',    'notcsv.txt',   SystemExit),

        ('hidden_features', '',     SystemExit),
        ('hidden_features', ' ',    SystemExit),
        ('hidden_features', '50',   SystemExit),
        ('hidden_features', '10 20 30 40 50 60',  SystemExit),
        ('hidden_features', '1 1 1',            SystemExit),
        ('hidden_features', '0 50 50',          SystemExit),
        ('hidden_features', '-1 50 50',         SystemExit),
        ('hidden_features', '10.0 20.0 30.0',   SystemExit),
        ('hidden_features', '501 10 10',        SystemExit),
        ('hidden_features', 'abc',              SystemExit),
        ('hidden_features', '10 20 30 40 50a',  SystemExit),
        ('hidden_features', '10 20 30 40 nan',  SystemExit),

        ('learning_rate',   '',     SystemExit),
        ('learning_rate',   ' ',    SystemExit),
        ('learning_rate',   '0',    SystemExit),
        ('learning_rate',   '-0.1', SystemExit),
        ('learning_rate',   '1.2',  SystemExit),
        ('learning_rate',   'inf',  SystemExit),
        ('learning_rate',   'nan',  SystemExit),

        ('weight_decay',    '',     SystemExit),
        ('weight_decay',    ' ',    SystemExit),
        ('weight_decay',    '-0.1', SystemExit),
        ('weight_decay',    '1.2',  SystemExit),
        ('weight_decay',    'inf',  SystemExit),
        ('weight_decay',    'nan',  SystemExit),

        ('optimizer',       '',     SystemExit),
        ('optimizer',       ' ',    SystemExit),
        ('optimizer',       'x',    SystemExit),

        ('verbose',         '',     SystemExit),
        ('verbose',         ' ',    SystemExit),
        ('verbose',         'yes',  SystemExit),

        ('plot',            '',     SystemExit),
        ('plot',            ' ',    SystemExit),
        ('plot',            'yes',  SystemExit),

        ('metrics_interval',   '',      SystemExit),
        ('metrics_interval',   '0',     SystemExit),
        ('metrics_interval',   '-100',  SystemExit),
        ('metrics_interval',   '-1000', SystemExit),
        ('metrics_interval',   'nan',   SystemExit),
        ('metrics_interval',   'inf',   SystemExit),

        ('save_dir',    None,       SystemExit),
        ('save_dir',    'nothing',  SystemExit), ])
    def test_invalid_arguments(self, field, value, expected_error):
        invalid_args = self.base_args.copy()
        invalid_args[field] = value
        sys.argv = dict_to_argv(self.filename, invalid_args)
        with pytest.raises(expected_error):
            train.parse_arguments()


class TestPredictArgument:
    @classmethod
    def setup_class(cls):
        cls.base_args = {
            'model_path':       'data/model.pkl',
            'dataset_path':     'data/data_test.csv',
        }
        cls.filename = 'predict.py'

    def test_train_arguments(self):
        sys.argv = dict_to_argv(self.filename, self.base_args)
        args = predict.parse_arguments()
        assert args.model_path == 'data/model.pkl'
        assert args.dataset_path == 'data/data_test.csv'

    @pytest.mark.parametrize("field, value, expected_error", [
        ('model_path',      None,           SystemExit),
        ('model_path',      '',             SystemExit),
        ('model_path',      ' ',            SystemExit),
        ('model_path',      'nothing',      SystemExit),
        ('model_path',      'notpkl.npz',   SystemExit),

        ('dataset_path',    None,           SystemExit),
        ('dataset_path',    '',             SystemExit),
        ('dataset_path',    ' ',            SystemExit),
        ('dataset_path',    'nothing',      SystemExit),
        ('dataset_path',    'notcsv.txt',   SystemExit), ])
    def test_invalid_arguments(self, field, value, expected_error):
        invalid_args = self.base_args.copy()
        invalid_args[field] = value
        sys.argv = dict_to_argv(self.filename, invalid_args)
        with pytest.raises(expected_error):
            predict.parse_arguments()

import pytest
import sys
import numpy as np
from srcs import dataloader, train, predict


def dict_to_argv(file, arg_dict):
    argv = [file]
    for key, value in arg_dict.items():
        if value is not None:
            argv.extend([f'--{key}', str(value)])
    return argv


class TestDataloaderArgument:
    @classmethod
    def setup_class(cls):
        cls.base_args = {
            'dataset_path': '../data/data.csv',
            'train_size': '0.8',
            'shuffle': 'true',
            'save_npz': 'false',
            'save_dir': '../data'
        }
        cls.filename = 'dataloader.py'

    def test_dataloader_arguments(self):
        sys.argv = dict_to_argv(self.filename, self.base_args)
        args = dataloader.parse_arguments()
        assert args.dataset_path == '../data/data.csv'
        assert args.train_size == 0.8
        assert args.shuffle
        assert not args.save_npz
        assert args.save_dir == '../data'

    @pytest.mark.parametrize("field, value, expected_error", [
        ('dataset_path',  None,         SystemExit),
        ('dataset_path', '',            SystemExit),
        ('dataset_path', 'nothing',     SystemExit),
        ('dataset_path', 'notcsv.npz',  SystemExit),

        ('train_size',  '',     SystemExit),
        ('train_size',  '0',    SystemExit),
        ('train_size',  '-0.1', SystemExit),
        ('train_size',  '1.2',  SystemExit),
        ('train_size',  'inf',  SystemExit),
        ('train_size',  'nan',  SystemExit),

        ('shuffle',     '',     SystemExit),

        ('save_npz',    'yes',  SystemExit),

        ('save_dir',    None,  SystemExit), ])
    def test_invalid_arguments(self, field, value, expected_error):
        invalid_args = self.base_args.copy()
        invalid_args[field] = value
        sys.argv = dict_to_argv(self.filename, invalid_args)
        with pytest.raises(expected_error):
            dataloader.parse_arguments()

    @pytest.mark.parametrize("field, value, expected", [
        ('train_size', '0.5', 0.5),
        ('shuffle', 'false', False),
        ('save_npz', 'true', True), ])
    def test_valid_argument_variations(self, field, value, expected):
        valid_args = self.base_args.copy()
        valid_args[field] = value
        sys.argv = dict_to_argv(self.filename, valid_args)
        args = dataloader.parse_arguments()
        assert getattr(args, field) == expected

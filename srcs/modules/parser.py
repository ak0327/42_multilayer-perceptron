import argparse
import numpy as np


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def str_expected(expected_strs: list[str]):
    lower_list = [s.upper() for s in expected_strs]
    def _checker(arg):
        s = arg.upper()
        if s not in lower_list:
            raise argparse.ArgumentTypeError(f"{arg} is not expected: {lower_list}")
        return s
    return _checker


def validate_extention(expected_ext: list[str]):
    extensions = [ext.lower() for ext in expected_ext]
    def _checker(arg):
        filename = arg.lower()
        if any(filename.endswith(ext) for ext in extensions):
            return arg
        raise argparse.ArgumentTypeError(f"{arg} is not expected: {extensions}")
    return _checker


def int_range(min_val, max_val):
    def _checker(arg):
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{arg} is not a valid integer")
        if value < min_val or max_val < value:
            raise argparse.ArgumentTypeError(f"{value} is not in range"
                                             f" [{min_val}, {max_val}]")
        return value
    return _checker


def float_range(min_val, max_val):
    def _checker(arg):
        try:
            value = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{arg} is not a valid float")
        if np.isnan(value):
            raise argparse.ArgumentTypeError(f"{arg} is NaN")
        if value < min_val or max_val < value:
            raise argparse.ArgumentTypeError(f"{value} is not in range"
                                             f" [{min_val}, {max_val}]")
        return value
    return _checker


def float_range_exclusive(min_val, max_val):
    def _checker(arg):
        try:
            value = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{arg} is not a valid float")
        if np.isnan(value):
            raise argparse.ArgumentTypeError(f"{arg} is NaN")
        if value <= min_val or max_val <= value:
            raise argparse.ArgumentTypeError(f"{value} is not in range"
                                             f" ({min_val}, {max_val})")
        return value
    return _checker


def int_list_type(min_elements, max_elements, min_value, max_value):
    def _checker(arg):
        if isinstance(arg, str):
            values = arg.split()
        elif isinstance(arg, list):
            values = arg
        else:
            raise argparse.ArgumentTypeError(f"Invalid input type for hidden_features")

        try:
            values = [int(v) for v in values]
        except ValueError:
            raise argparse.ArgumentTypeError(f"All elements of hidden_features must be integers")

        if not (min_elements <= len(values) <= max_elements):
            raise argparse.ArgumentTypeError(f"Must have {min_elements} to {max_elements} elements")

        if any(value < min_value or max_value < value for value in values):
            raise argparse.ArgumentTypeError(f"All values in hidden_features must be between {min_value} and {max_value}")

        return values
    return _checker

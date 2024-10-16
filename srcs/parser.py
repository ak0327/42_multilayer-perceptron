import argparse


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def int_range(arg, min_val, max_val):
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg} is not a valid integer")
    if value < min_val or max_val < value:
        raise argparse.ArgumentTypeError(f"{value} is not in range"
                                         f" [{min_val}, {max_val}]")
    return value


def float_range(arg, min_val, max_val):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg} is not a valid float")
    if value < min_val or max_val < value:
        raise argparse.ArgumentTypeError(f"{value} is not in range"
                                         f" [{min_val}, {max_val}]")
    return value


def float_range_exclusive(arg, min_val, max_val):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg} is not a valid float")
    if value <= min_val or max_val <= value:
        raise argparse.ArgumentTypeError(f"{value} is not in range"
                                         f" ({min_val}, {max_val})")
    return value

import numpy as np


def he_normal(in_features: int, out_features: int, seed: int = None) -> np.ndarray:
    """
    He initialization for neural network weights.
    This method is designed for layers with ReLU activation.

    :param in_features: Size of the input dimension.
    :param out_features: Size of the output dimension.
    :returns: Initialized weights.
    """
    if in_features == 0:
        raise ValueError("Input dimension cannot be zero.")

    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)


def xavier_normal(in_features: int, out_features: int, seed: int = None) -> np.ndarray:
    """
    Xavier (Glorot) initialization for neural network weights.
    This method is designed for layers with tanh or sigmoid activation.

    :param in_features: Size of the input dimension.
    :param out_features: Size of the output dimension.
    :returns: Initialized weights.
    """
    if in_features == 0:
        raise ValueError("Input dimension cannot be zero.")

    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)


def normal(in_features: int, out_features: int, std: float = 0.01, seed: int = None) -> np.ndarray:
    """
    Normal (Gaussian) initialization for neural network weights.

    :param in_features: Size of the input dimension.
    :param out_features: Size of the output dimension.
    :param std: Standard deviation of the normal distribution. Default is 0.01.
    :returns: Initialized weights.
    """
    if in_features == 0:
        raise ValueError("Input dimension cannot be zero.")
    if std < 0:
        raise ValueError("Standard deviation must be non-negative.")

    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(in_features, out_features) * std

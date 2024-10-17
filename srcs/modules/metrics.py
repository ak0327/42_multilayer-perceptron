import numpy as np


def _one_hot_encoding(y_true, y_pred):
    # Convert probabilities to class predictions if necessary
    if y_pred.ndim != 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Convert one-hot encoded true labels to class labels if necessary
    if y_true.ndim != 1:
        y_true = np.argmax(y_true, axis=1)

    return y_true, y_pred


def _get_confusion_matrix(y_true, y_pred):
    _assert_sample_size(y_true, y_pred)
    y_true, y_pred = _one_hot_encoding(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn


def _assert_sample_size(y_true: np.ndarray, y_pred: np.ndarray):
    t_size = len(y_true)
    y_size = len(y_pred)
    if t_size != y_size:
        raise ValueError(f"Found input variables with inconsistent numbers "
                         f"of samples: [{t_size}, {y_size}]")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp, fp, fn, tn = _get_confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division=0.0):
    tp, fp, fn, tn = _get_confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else zero_division
    return precision


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division=0.0):
    tp, fp, fn, tn = _get_confusion_matrix(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else zero_division
    return recall


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division=0.0):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else zero_division
    return f1


def get_metrics(y, t):
    accuracy = accuracy_score(y_true=t, y_pred=y)
    precision = precision_score(y_true=t, y_pred=y)
    recall = recall_score(y_true=t, y_pred=y)
    f1 = f1_score(y_true=t, y_pred=y)
    return accuracy, precision, recall, f1

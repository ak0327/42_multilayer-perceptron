import pytest
import numpy as np
from sklearn import metrics as sk

from srcs.modules import metrics as ft


class TestMetrics:
    def _assert_score(self, t, y, case_id):
        expected_accuracy = sk.accuracy_score(t, y)
        expected_precision = sk.precision_score(t, y, zero_division=0.0)
        expected_recall = sk.recall_score(t, y, zero_division=0.0)
        expected_f1 = sk.f1_score(t, y, zero_division=0.0)

        actual_accuracy = ft.accuracy_score(t, y)
        actual_precision = ft.precision_score(t, y, zero_division=0.0)
        actual_recall = ft.recall_score(t, y, zero_division=0.0)
        actual_f1 = ft.f1_score(t, y, zero_division=0.0)

        assert expected_accuracy == actual_accuracy, f"Accuracy: {case_id}"
        assert expected_precision == actual_precision, f"Precision: {case_id}"
        assert expected_recall == actual_recall, f"Recall: {case_id}"
        assert expected_f1 == actual_f1, f"F1: {case_id}"


    @pytest.mark.parametrize("y_true, y_pred, case_id", [
        ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], "empty"),
        ([0, 1], [1, 1], "simple binary"),
    ])
    def test_valid(self, y_true, y_pred, case_id):
        t = np.array(y_true)
        y = np.array(y_pred)
        self._assert_score(t=t, y=y, case_id=case_id)

    # @pytest.mark.parametrize("y_true, y_pred, case_id", [
    #     ([], [], "empty"),
    #     # ([0, 1], [1, 1, 0], "different size"),
    # ])
    # def test_invalid(self, y_true, y_pred, case_id):
    #     t = np.array(y_true)
    #     y = np.array(y_pred)
    #     self._assert_score(t=t, y=y, case_id=case_id)

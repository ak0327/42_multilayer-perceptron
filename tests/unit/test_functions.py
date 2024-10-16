import pytest
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim

from srcs.modules.functions import Softmax


class TestSoftmax:
    def _assert_softmax(self, x):
        # original
        softmax = Softmax()
        actual = softmax(x)

        # PyTorch
        x_torch = torch.tensor(x, dtype=torch.float32)
        dim = 0 if x_torch.dim() == 1 else -1
        expected = F.softmax(x_torch, dim=dim).numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("x", [
        np.array([0, 0, 0]),
        np.array([0.1, 0.2, 0.7]),
        np.array([1000, 10000, 100000]),
        np.array([-1000, -10000, -100000]),
        np.array([np.finfo(np.float32).max, np.finfo(np.float32).max]),
        np.array([-1e10, -1e100, -1e200]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        np.array([1e-300, 1e-200, 1e-100]),
    ])
    def test_valid_x(self, x):
        self._assert_softmax(x)

    @pytest.mark.parametrize("x", [
        np.arange(100),
        np.random.rand(1000),
        np.arange(10000).reshape(100, 100),
        np.arange(1000000).reshape(1000, 1000),
    ])
    def test_large_dim_x(self, x):
        self._assert_softmax(x)

    @pytest.mark.parametrize("x", [
        np.array([1, 0, np.nan]),
        np.array([1, 0, np.inf]),
        np.array([np.nan, np.nan, np.nan]),
        np.array([np.inf, np.inf, np.inf]),
        np.array([-np.inf, -np.inf, -np.inf])])
    def test_special_x(self, x):
        self._assert_softmax(x)

    def test_empty(self):
        x = np.array([])
        softmax = Softmax()
        result = softmax(x)
        assert result.size == 0, "Result should be an empty array"

        x_torch = torch.tensor([], dtype=torch.float32)
        torch_result = F.softmax(x_torch, dim=0).numpy()
        assert torch_result.size == 0, "PyTorch result should be an empty array"

        np.testing.assert_array_equal(result, torch_result)

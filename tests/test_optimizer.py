import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from srcs.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam


class TestSGD:
    def _assert_update(self, lr, params, grads):
        original_error = None
        torch_error = None

        # original SGD
        try:
            original_sgd = SGD(lr=lr)
            original_params = {k: v.copy() for k, v in params.items()}
            original_grads = {k: v.copy() for k, v in grads.items()}
            original_sgd.update(original_params, original_grads)
        except Exception as e:
            original_error = type(e)

        # PyTorch SGD
        try:
            torch_params = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
            torch_grads = {k: torch.tensor(v) for k, v in grads.items()}
            torch_sgd = optim.SGD(torch_params.values(), lr=lr)
            torch_sgd.zero_grad()
            for key in torch_params:
                torch_params[key].grad = torch_grads[key]
            torch_sgd.step()
        except Exception as e:
            torch_error = type(e)

        if original_error is None and torch_error is None:
            # compare
            for key in params:
                np.testing.assert_allclose(
                    original_params[key],
                    torch_params[key].detach().numpy(),
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Mismatch in parameter {key}"
                )
        else:
            # Exception handling
            if type(original_error) != type(torch_error):
                pytest.fail(
                    f"Inconsistent error handling: "
                    f"Original SGD raised {type(original_error)}, "
                    f"PyTorch SGD raised {type(torch_error)}"
                )
            elif original_error is None:
                pytest.fail(
                    "Original SGD did not raise an error, but PyTorch SGD did"
                )
            elif torch_error is None:
                pytest.fail(
                    "PyTorch SGD did not raise an error, but Original SGD did"
                )
            else:
                raise original_error

    def test_sgd_single_param(self):
        lr = 0.1
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        1e-10, 0.001, 0.01, 0.1, 1.0, 1e10, np.finfo(np.float32).max
    ])
    def test_sgd_invalid_learning_rates(self, lr):
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        0, np.nan, np.inf
    ])
    def test_sgd_invalid_learning_rates(self, lr):
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        np.finfo(np.float64).min,   # 最小の負の値
        -np.inf,                    # 負の無限大
        -1,                         # 通常の負の値
        np.nextafter(0, -1),        # ゼロに最も近い負の値
    ])
    def test_sgd_invalid_learning_rates(self, lr):
        with pytest.raises(ValueError):
            params = {'w': np.array([1.0, 2.0, 3.0])}
            grads = {'w': np.array([0.1, 0.2, 0.3])}
            self._assert_update(lr, params, grads)

    def test_sgd_multiple_params(self):
        lr = 0.01
        params = {
            'w1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'w2': np.array([0.5, 0.6])
        }
        grads = {
            'w1': np.array([[0.01, 0.02], [0.03, 0.04]]),
            'w2': np.array([0.05, 0.06])
        }
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("params, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params"),
    ])
    def test_sgd_params(self, params, case_id):
        lr = 0.1
        grads = [1.0, 1.0, 1.0]
        self._assert_update(
            lr=lr,
            params={'w': np.array(params)},
            grads={'w': np.array(grads)}
        )

    @pytest.mark.parametrize("grads, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params"),
    ])
    def test_sgd_grads(self, grads, case_id):
        lr = 0.1
        params = [1.0, 2.0, 3.0]
        self._assert_update(
            lr=lr,
            params={'w': np.array(params)},
            grads={'w': np.array(grads)}
        )

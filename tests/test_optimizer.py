import pytest
import numpy as np
from typing import Any

import torch
import torch.nn as nn
from torch import optim

from srcs.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam


class TestOptimizers:
    def _assert_update(
            self,
            original_optimizer_class: type,
            torch_optimizer_class: type,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            original_raise: bool = False,
            optimizer_kwargs: dict[str, Any] = {}
    ):
        original_error = None
        torch_error = None

        # Original optimizer
        try:
            original_optimizer = original_optimizer_class(lr=lr, **optimizer_kwargs)
            original_params = {k: v.copy() for k, v in params.items()}
            original_grads = {k: v.copy() for k, v in grads.items()}
            original_optimizer.update(original_params, original_grads)
        except Exception as e:
            original_error = e
            print(f"Original optimizer error: {e}")

        # PyTorch optimizer
        try:
            torch_params = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
            torch_grads = {k: torch.tensor(v) for k, v in grads.items()}
            torch_optimizer = torch_optimizer_class(torch_params.values(), lr=lr, **optimizer_kwargs)
            torch_optimizer.zero_grad()
            for key in torch_params:
                torch_params[key].grad = torch_grads[key]
            torch_optimizer.step()
        except Exception as e:
            torch_error = e
            print(f"PyTorch optimizer error: {e}")

        if original_raise and original_error:
            raise original_error

        if original_error is None and torch_error is None:
            # Compare results
            for key in params:
                np.testing.assert_allclose(
                    original_params[key],
                    torch_params[key].detach().numpy(),
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Mismatch in parameter {key}"
                )
        else:
            # Compare the exception types and messages
            assert type(original_error) == type(torch_error), \
                f"Inconsistent error types: " \
                f"Original optimizer raised {type(original_error)}," \
                f" PyTorch optimizer raised {type(torch_error)}"

            assert str(original_error) == str(torch_error), \
                f"Inconsistent error messages: " \
                f"Original optimizer message: {original_error}, " \
                f"PyTorch optimizer message: {torch_error}"


class TestSGD(TestOptimizers):
    def _assert_update(
            self,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            optimizer_kwargs: dict[str, Any] = {}
    ):
        super()._assert_update(
            original_optimizer_class=SGD,
            torch_optimizer_class=optim.SGD,
            lr=lr,
            params=params,
            grads=grads,
            optimizer_kwargs=optimizer_kwargs
        )

    def test_single_param(self):
        lr = 0.1
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        1e-10, 0.001, 0.01, 0.1, 1.0, 1e10, np.finfo(np.float32).max])
    def test_valid_lr(self, lr):
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        0, np.nan, np.inf])
    def test_special_lr(self, lr):
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("lr", [
        np.finfo(np.float64).min,   # 最小の負の値
        -np.inf,                    # 負の無限大
        -1,                         # 通常の負の値
        np.nextafter(0, -1)])      # ゼロに最も近い負の値
    def test_invalid_lr(self, lr):
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    def test_multiple_params(self):
        lr = 0.01
        params = {
            'w1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'w2': np.array([0.5, 0.6]),
            'w3': np.array([-0.2, 1.5])
        }
        grads = {
            'w1': np.array([[0.01, 0.02], [0.03, 0.04]]),
            'w2': np.array([0.05, 0.06]),
            'w3': np.array([-0.02, 0.09])
        }
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("params, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params")])
    def test_params(self, params, case_id):
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
        ([0.0, 0.0, 0.0]        , "zero_params"),])
    def test_grads(self, grads, case_id):
        lr = 0.1
        params = [1.0, 2.0, 3.0]
        self._assert_update(
            lr=lr,
            params={'w': np.array(params)},
            grads={'w': np.array(grads)}
        )


class TestMomentum(TestSGD):
    def _assert_update(
            self,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            momentum: float = 0.0,
            nesterov: bool = False
    ):
        TestOptimizers._assert_update(
            self,
            original_optimizer_class=Momentum,
            torch_optimizer_class=optim.SGD,
            lr=lr,
            params=params,
            grads=grads,
            optimizer_kwargs={'momentum': momentum, 'nesterov': nesterov}
        )

    def test_single_param(self):
        lr = 0.1
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads)

    @pytest.mark.parametrize("momentum", [
        1e-10, 0.001, 0.01, 0.1, 1.0, 1e10, np.finfo(np.float32).max])
    def test_invalid_momentum(self, momentum):
        lr = 0.01
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads, momentum)

    @pytest.mark.parametrize("momentum", [
        0, np.nan, np.inf])
    def test_special_momentum(self, momentum):
        lr = 0.01
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads, momentum)

    @pytest.mark.parametrize("momentum", [
        np.finfo(np.float64).min,   # 最小の負の値
        -np.inf,                    # 負の無限大
        -1,                         # 通常の負の値
        np.nextafter(0, -1)])       # ゼロに最も近い負の値
    def test_momentum_invalid_momentum(self, momentum):
        lr = 0.01
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads, momentum)

    @pytest.mark.parametrize("params, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params")])
    def test_params(self, params, case_id):
        lr = 0.1
        momentums = [1e-10, 0.01, 1e10]
        grads = [1.0, 1.0, 1.0]
        for momentum in momentums:
            self._assert_update(
                lr=lr,
                params={'w': np.array(params)},
                grads={'w': np.array(grads)},
                momentum=momentum
            )

    @pytest.mark.parametrize("grads, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params")])
    def test_grads(self, grads, case_id):
        lr = 0.1
        momentums = [1e-10, 0.01, 1e10]
        params = [1.0, 2.0, 3.0]
        for momentum in momentums:
            self._assert_update(
                lr=lr,
                params={'w': np.array(params)},
                grads={'w': np.array(grads)},
                momentum=momentum
            )


class TestNesterov(TestMomentum):
    def _assert_update(
            self,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            momentum: float = 0.0
    ):
        super()._assert_update(
            lr=lr,
            params=params,
            grads=grads,
            momentum=momentum,
            nesterov=True
        )

    @pytest.mark.parametrize("momentum", [
        0,
        np.nan,
        # np.inf  # skip
    ])
    def test_special_momentum(self, momentum):
        lr = 0.01
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads, momentum)


class TestAdaGrad(TestSGD):
    def _assert_update(
            self,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            optimizer_kwargs: dict[str, Any] = {}
    ):
        TestOptimizers._assert_update(
            self,
            original_optimizer_class=AdaGrad,
            torch_optimizer_class=optim.Adagrad,
            lr=lr,
            params=params,
            grads=grads,
            optimizer_kwargs=optimizer_kwargs
        )

    @pytest.mark.parametrize("grads, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        # ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params")])
    def test_grads(self, grads, case_id):
        lr = 0.1
        params = [1.0, 2.0, 3.0]
        self._assert_update(
            lr=lr,
            params={'w': np.array(params)},
            grads={'w': np.array(grads)}
        )


class TestRMSProp(TestSGD):
    def _assert_update(
            self,
            lr: float,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray],
            original_raise: bool = False,
            alpha: float = 0.99
    ):
        TestOptimizers._assert_update(
            self,
            original_optimizer_class=RMSProp,
            torch_optimizer_class=optim.RMSprop,
            lr=lr,
            params=params,
            grads=grads,
            original_raise=original_raise,
            optimizer_kwargs={'alpha': alpha}
        )

    @pytest.mark.parametrize("grads, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        # ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params")])
    def test_grads(self, grads, case_id):
        lr = 0.1
        params = [1.0, 2.0, 3.0]
        self._assert_update(
            lr=lr,
            params={'w': np.array(params)},
            grads={'w': np.array(grads)}
        )

    @pytest.mark.parametrize("alpha", [
        0.0, 1e-100, 1e-10, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99999
    ])
    def test_rmsprop_alpha_less_than_1(self, alpha):
        lr = 0.01
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}
        self._assert_update(lr, params, grads, alpha)

    @pytest.mark.parametrize("alpha", [
        np.finfo(np.float64).min,   # 最小の負の値
        -np.inf,                    # 負の無限大
        -1,                         # 通常の負の値
        np.nextafter(0, -1),        # ゼロに最も近い負の値
        1.0,                        # 1.0以上の値
        10.0,
        np.inf,
        np.nan,
    ])
    def test_rmsprop_invalid_alpha(self, alpha):
        with pytest.raises(ValueError):
            original_raise = True
            lr = 0.01
            params = {'w': np.array([1.0, 2.0, 3.0])}
            grads = {'w': np.array([0.1, 0.2, 0.3])}
            self._assert_update(lr, params, grads, original_raise, alpha)

    @pytest.mark.parametrize("params, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params"),
    ])
    def test_rmsprop_params(self, params, case_id):
        lr = 0.1
        alphas = [0, 1e-100, 1e-10, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.99999]
        grads = [1.0, 1.0, 1.0]
        for alpha in alphas:
            self._assert_update(
                lr=lr,
                params={'w': np.array(params)},
                grads={'w': np.array(grads)},
                alpha=alpha
            )

    @pytest.mark.parametrize("grads, case_id", [
        ([1e10, 1e15, 1e20]     , "very_large_values"),
        ([1e-10, 1e-15, 1e-20]  , "very_small_values"),
        ([1e-10, 1.0, 1e10]     , "mixed_large_and_small_values"),
        # ([1.0, np.inf, -np.inf] , "inf_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([1.0, np.nan, 3.0]     , "nan_values"),
        ([0.0, 0.0, 0.0]        , "zero_params"),
    ])
    def test_rmsprop_grads(self, grads, case_id):
        lr = 0.1
        alphas = [0, 1e-100, 1e-10, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.99999]
        params = [1.0, 2.0, 3.0]
        for alpha in alphas:
            self._assert_update(
                lr=lr,
                params={'w': np.array(params)},
                grads={'w': np.array(grads)},
                alpha=alpha
            )

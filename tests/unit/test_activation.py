import pytest
import numpy as np
import torch
import torch.nn as nn

from srcs.modules.activation import ReLU, Sigmoid


class TestReLU:
    def _assert_forward(self, x, expected_output):
        _relu = ReLU()
        _actual_output = _relu(x)
        np.testing.assert_array_equal(_actual_output, expected_output)

        _torch_relu = nn.ReLU()
        _torch_x = torch.tensor(x, dtype=torch.float32)
        _torch_output = _torch_relu(_torch_x)
        np.testing.assert_allclose(_actual_output, _torch_output, rtol=1e-5, atol=1e-8)

    def _assert_backward(self, x):
        _dout = np.arange(len(x), dtype=np.float32)
        _expected_grad = np.where(0 < x, _dout, 0)

        _relu = ReLU()
        _relu(x)
        _actual_grad = _relu.backward(_dout)
        np.testing.assert_array_equal(_actual_grad, _expected_grad)

        _torch_relu = nn.ReLU()
        _torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        _torch_dout = torch.tensor(_dout, dtype=torch.float32)
        _torch_out = _torch_relu(_torch_x)
        _torch_out.backward(_torch_dout)
        _torch_grad = _torch_x.grad.numpy()

        np.testing.assert_allclose(_actual_grad, _torch_grad, rtol=1e-5, atol=1e-8)

    def test_forward(self):
        x               = np.array([-2, -1, 0, 0.01, 1, 2])
        expected_output = np.array([ 0,  0, 0, 0.01, 1, 2])
        self._assert_forward(x, expected_output)

    def test_forward_float_limits(self):
        x               = np.array([np.finfo(np.float32).min, np.finfo(np.float32).max])
        expected_output = np.array([0.0                     , np.finfo(np.float32).max])
        self._assert_forward(x, expected_output)

    def test_forward_inf(self):
        x               = np.array([-np.inf, np.inf])
        expected_output = np.array([0.0    , np.inf])
        self._assert_forward(x, expected_output)

    def test_forward_nan(self):
        x = np.array([np.nan, -1, 0, 1])
        expected_output = np.array([np.nan, 0, 0, 1])
        self._assert_forward(x, expected_output)

    def test_backward(self):
        x = np.array([-2, -1, 0, 1, 2])
        self._assert_backward(x)

    def test_backward_float_limits(self):
        x = np.array([np.finfo(np.float32).min, np.finfo(np.float32).max])
        self._assert_backward(x)

    def test_backward_inf(self):
        x = np.array([-np.inf, np.inf])
        self._assert_backward(x)

    def test_backward_nan(self):
        x = np.array([np.nan, -1, 0, 1])
        self._assert_backward(x)


class TestSigmoid:
    def _assert_forward(self, x):
        _relu = Sigmoid()
        _actual_output = _relu(x)

        _torch_sigmoid = nn.Sigmoid()
        _torch_x = torch.tensor(x, dtype=torch.float32)
        _torch_output = _torch_sigmoid(_torch_x)
        np.testing.assert_allclose(_actual_output, _torch_output, rtol=1e-5, atol=1e-8)

    def _assert_backward(self, x):
        _dout = np.arange(len(x), dtype=np.float32)

        _relu = Sigmoid()
        _relu(x)
        _actual_grad = _relu.backward(_dout)

        _torch_sigmoid = nn.Sigmoid()
        _torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        _torch_dout = torch.tensor(_dout, dtype=torch.float32)
        _torch_out = _torch_sigmoid(_torch_x)
        _torch_out.backward(_torch_dout)
        _torch_grad = _torch_x.grad.numpy()

        np.testing.assert_allclose(_actual_grad, _torch_grad, rtol=1e-5, atol=1e-8)

    def test_forward(self):
        x = np.array([-1, 0, 1])
        self._assert_forward(x)

    def test_forward_float_limits(self):
        x = np.array([np.finfo(np.float32).min, np.finfo(np.float32).max])
        self._assert_forward(x)

    def test_forward_inf(self):
        x = np.array([-np.inf, np.inf])
        self._assert_forward(x)

    def test_forward_nan(self):
        x = np.array([np.nan])
        self._assert_forward(x)

    def test_backward(self):
        x = np.array([-1, 0, 1])
        self._assert_backward(x)

    def test_backward_float_limits(self):
        x = np.array([np.finfo(np.float32).min, np.finfo(np.float32).max])
        self._assert_backward(x)

    def test_backward_inf(self):
        x = np.array([-np.inf, np.inf])
        self._assert_backward(x)

    def test_backward_nan(self):
        x = np.array([np.nan])
        self._assert_backward(x)

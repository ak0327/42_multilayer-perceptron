import pytest
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim

# from srcs.loss import cross_entropy
# from srcs.functions import Softmax
#
#
# class TestCrossEntropy:
#     def _assert_cross_entropy(self, logit, target):
#         # original
#         y = softmax(logit)
#         actual = cross_entropy(y, target)
#
#         # PyTorch
#         logit_torch = torch.from_numpy(logit).float()
#
#         if target.ndim > 1:
#             # ワンホットエンコーディングをクラスインデックスに変換
#             t_torch = torch.from_numpy(target).argmax(dim=1).long()
#         else:
#             # 1次元のワンホットエンコーディングをクラスインデックスに変換
#             t_torch = torch.from_numpy(target).argmax().long()
#
#         # バッチ次元の追加（logit が1次元の場合）
#         if logit_torch.dim() == 1:
#             logit_torch = logit_torch.unsqueeze(0)  # (1, C) に変更
#             t_torch = t_torch.unsqueeze(0)          # (1,) に変更
#
#         loss = F.cross_entropy(logit_torch, t_torch)
#         expected = loss.item()
#
#         assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8), \
#             f"Mismatch for input y={y}, target={target}. Got {actual}, expected {expected}"
#
#     @pytest.mark.parametrize("logit, target, case_id", [
#         (np.array([0.1, 0.2, 0.7])                      , np.array([0, 0, 1])               , "1d_input"),
#         (np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])   , np.array([[0, 0, 1], [1, 0, 0]])  , "2d_input_one_hot"),
#     ])
#     def test_cross_entropy(self, logit, target, case_id):
#         self._assert_cross_entropy(logit, target)

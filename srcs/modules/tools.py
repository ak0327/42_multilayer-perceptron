import copy
import numpy as np
from srcs.modules.model import Sequential


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = max(0, patience)
        self.verbose = verbose
        self.counter = 0
        self.prev_loss = np.Inf
        self.early_stop = False
        self.best_model = None

    def __call__(self, current_valid_loss, model: Sequential):
        if self.patience == 0:
            return

        if current_valid_loss < self.prev_loss:
            # 前epochのloss以下の場合
            self.prev_loss = current_valid_loss
            self.counter = 0
            self._checkpoint(current_valid_loss, model)
        else:
            # 前epochのlossよりも大きくなった場合
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: '
                      f'{self.counter}/{self.patience}')
            if self.patience <= self.counter:
                if self.verbose:
                    print(f'Ealry stopping with best loss: {self.prev_loss}')
                self.early_stop = True

    def _checkpoint(self, current_valid_loss, model: Sequential):
        if self.verbose:
            print(f'Validation loss decreased '
                  f'({self.prev_loss:.6f} -> {current_valid_loss:.6f}). '
                  f'Saving model...')
        self.best_model = copy.deepcopy(model)

# class EarlyStopping:
#     def __init__(self, patience=5, verbose=False, min_delta=1e-4):
#         self.patience = max(0, patience)
#         self.verbose = verbose
#         self.counter = 0
#         self.prev_loss = np.Inf
#         self.early_stop = False
#         self.best_model = None
#         self.min_delta = min_delta  # 最小の改善幅を指定
#
#     def __call__(self, current_valid_loss, model: Sequential):
#         if self.patience == 0:
#             return
#
#         # 前回の損失からの改善幅を確認
#         loss_improvement = self.prev_loss - current_valid_loss
#
#         if loss_improvement > self.min_delta:
#             # 閾値以上の改善があった場合にのみモデルを保存
#             self.prev_loss = current_valid_loss
#             self.counter = 0
#             self._checkpoint(current_valid_loss, model)
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter}/{self.patience}')
#             if self.patience <= self.counter:
#                 if self.verbose:
#                     print(f'Early stopping with best loss: {self.prev_loss}')
#                 self.early_stop = True
#
#     def _checkpoint(self, current_valid_loss, model: Sequential):
#         if self.verbose:
#             print(f'Validation loss decreased '
#                   f'({self.prev_loss:.6f} -> {current_valid_loss:.6f}). '
#                   f'Saving model...')
#         self.best_model = copy.deepcopy(model)

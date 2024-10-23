import copy
import numpy as np
import pandas as pd
from srcs.modules.model import Sequential


def normalize(df: pd.DataFrame, columns: list[str]):
    # df_norm = df.copy()
    # for col in columns:
    #     min_val = df[col].min()
    #     max_val = df[col].max()
    #     range_val = max_val - min_val
    #     if range_val == 0:
    #         df_norm[col] = 1
    #     else:
    #         df_norm[col] = (df[col] - min_val) / range_val

    df_norm = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0:
            df_norm[col] = 0  # 標準偏差が0の場合、全て同じ値なので0
        else:
            df_norm[col] = (df[col] - mean_val) / std_val
    return df_norm


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

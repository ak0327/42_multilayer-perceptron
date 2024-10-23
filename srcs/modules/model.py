import numpy as np
from srcs.modules.functions import (
    Softmax, SoftmaxWithCrossEntropyLoss, numerical_gradient
)
from srcs.modules.activation import ReLU
from srcs.modules.loss import CrossEntropyLoss
from srcs.modules.optimizer import Optimizer
from srcs.modules.init import he_normal
from srcs.modules.layer import Affine, Dense, Linear
from srcs.modules.metrics import accuracy_score


class Sequential:
    def __init__(
            self,
            layers: list[Dense],
            criteria: callable,
            optimizer: Optimizer,
            weight_decay: float = 0.0,
            weight_init_std: float = 0.01,
    ):
        self._set_layers(layers)
        self.optimizer = optimizer
        self.criteria = criteria()
        self._set_sequential_info()
        self.weight_decay = weight_decay
        # 最終層の活性化関数がsoftmax, 損失関数がCrossEntropyLossの場合
        # 数値的安定化のため、損失関数をSoftmaxWithCrossEntropyLossに差し替える
        if (isinstance(self.layers[-1].activation, Softmax)
                and (isinstance(self.criteria, CrossEntropyLoss))):
            self.layers[-1].activation = None
            self.criteria = SoftmaxWithCrossEntropyLoss()

    def _set_layers(self, layers: list[Dense]):
        self.layers = layers
        for i, layer in enumerate(self.layers):
            layer.set_id(i)

    def _set_sequential_info(self):
        """
        最終層をSoftmaxWithCrossEntropyLossに変更する前に、infoを作成する

        Sequential(
          (0): Linear(in_features=8, out_features=12, bias=True)
          (1): ReLU()
          (2): Linear(in_features=12, out_features=8, bias=True)
          (3): ReLU()
          (4): Linear(in_features=8, out_features=1, bias=True)
          (5): Sigmoid()
        )
        """
        no = 0
        SP = " "
        info =  f"{SP * 1}MODEL:\n"
        info += f"{SP * 3}Sequential(\n"
        for layer in self.layers:
            linear, activation = layer.info()
            info += f"{SP * 5}({no}): {linear}\n"
            no += 1
            if activation is None:
                continue
            info += f"{SP * 5}({no}): {activation}\n"
            no += 1
        info += f"{SP * 3})\n"
        info += f"{SP * 1}OPTIMIZER: {self.optimizer.info}\n"
        info += f"{SP * 1}CRITERIA : {self.criteria.info}\n"
        self.sequential_info = info

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, y, t):
        loss = self.criteria(y, t)

        l2_reg = 0
        for layer in self.layers:
            l2_reg += (layer.W ** 2).sum()
        loss += 0.5 * self.weight_decay * l2_reg
        return loss

    def accuracy(self, y, t):
        return accuracy_score(y_true=t, y_pred=y)

    def backward(self):
        dout = self.criteria.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # L2正則化に基づく勾配を追加で計算して、パラメータに反映
        for layer in self.layers:
            if self.weight_decay > 0:
                dW = layer.dW + self.weight_decay * layer.W  # L2正則化の勾配を追加
                layer.set_dW(dW)

        grads = self.grads
        return grads

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(self.forward(x), t)

        numerical_grads = {}
        for i, layer in enumerate(self.layers):
            key_W = f'W_{i}'
            key_b = f'b_{i}'

            # 保存
            tmp_W = layer.W
            tmp_b = layer.b

            # backwardと同じ初期値でnumerical gradを計算
            layer.set_params(layer.init_W, layer.init_b)
            numerical_grads[key_W] = numerical_gradient(loss_W, layer.W)
            numerical_grads[key_b] = numerical_gradient(loss_W, layer.b)

            # 戻す
            layer.set_params(tmp_W, tmp_b)

        return numerical_grads

    def update_params(self):
        params = self.params
        grads = self.grads
        self.optimizer.update(params, grads)

        for i, layer in enumerate(self.layers):
            key_W = f'W_{i}'
            key_b = f'b_{i}'
            layer.set_params(W=params[key_W], b=params[key_b])

    @property
    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            key_W = f'W_{i}'
            key_b = f'b_{i}'
            params[key_W] = layer.W
            params[key_b] = layer.b
        return params

    @property
    def grads(self):
        grads = {}
        for i, layer in enumerate(self.layers):
            key_W = f'W_{i}'
            key_b = f'b_{i}'
            grads[key_W] = layer.dW
            grads[key_b] = layer.db
        return grads

    @property
    def info(self):
        return self.sequential_info

import numpy as np
from srcs.functions import Softmax, SoftmaxWithCrossEntropyLoss, numerical_gradient
from srcs.activation import ReLU
from srcs.loss import CrossEntropyLoss
from srcs.optimizer import Optimizer
from srcs.init import he_normal
from srcs.layer import Affine, Dense, Linear


class TwoLayerNet:
    def __init__(
            self,
            in_features: int,
            hid_features: int,
            out_features: int,
            optimizer: Optimizer,
            weight_init_std: float = 0.01,
    ):
        self.params = {
            'W1' : weight_init_std * np.random.randn(in_features, hid_features),
            'b1' : np.zeros(hid_features),
            'W2' : weight_init_std * np.random.randn(hid_features, out_features),
            'b2' : np.zeros(out_features),
        }
        self.layers = {
            'Affine1' : Affine(W=self.params['W1'], b=self.params['b1']),
            'Relu1'   : ReLU(),
            'Affine2' : Affine(W=self.params['W2'], b=self.params['b2']),
        }
        self.last_layer = SoftmaxWithCrossEntropyLoss()

        self.optimizer = optimizer

    def predict(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.last_layer(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backward(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {
            'W1' : self.layers['Affine1'].dW,
            'b1' : self.layers['Affine1'].db,
            'W2' : self.layers['Affine2'].dW,
            'b2' : self.layers['Affine2'].db,
        }
        return grads

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {
            'W1' : numerical_gradient(loss_W, self.params['W1']),
            'b1' : numerical_gradient(loss_W, self.params['b1']),
            'W2' : numerical_gradient(loss_W, self.params['W2']),
            'b2' : numerical_gradient(loss_W, self.params['b2']),
        }
        return grads

    def update_params(self, grads):
        self.optimizer.update(self.params, grads)


class Sequential:
    def __init__(
            self,
            layers: list[Dense],
            criteria: callable,
            optimizer: Optimizer,
            weight_init_std: float = 0.01,
    ):
        self.layers = layers
        self._set_layer_id()
        self._set_sequential_info()

        self.criteria = criteria()
        if (isinstance(self.layers[-1].activation, Softmax)
                and isinstance(self.criteria, CrossEntropyLoss)):
            self.layers[-1].activation = None
            self.criteria = SoftmaxWithCrossEntropyLoss()

        self.optimizer = optimizer

    def _set_layer_id(self):
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
        info = "Sequential(\n"
        for layer in self.layers:
            linear, activation = layer.info()
            info += f"  ({no}): {linear}\n"
            no += 1
            if activation is None:
                continue
            info += f"  ({no}): {activation}\n"
            no += 1
        info += ")"
        self.sequential_info = info

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.criteria(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backward(self, x, t):
        self.loss(x, t)

        dout = self.criteria.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        grads = self.grads
        return grads

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(x, t)

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

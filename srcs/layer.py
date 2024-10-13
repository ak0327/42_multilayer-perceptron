import numpy as np
from srcs.functions import Softmax, numerical_gradient
from srcs.activation import ReLU
from srcs.loss import cross_entropy
from srcs.optimizer import Optimizer
from srcs.init import he_normal


class Affine:
    def __init__(self, W, b):
        self.W: np.ndarray = W
        self.b: np.ndarray = b

        self.x: Optional[np.ndarray] = None
        self.x_shape = None

        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # テンソル対応
        self.x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class Linear:
    def __init__(
            self,
            in_features: int,
            out_features: int,
            init_method: callable = he_normal
    ):
        self.in_features = in_features
        self.out_features = out_features

        self.W: np.ndarray = init_method(in_features=in_features, out_features=out_features)
        self.b: np.ndarray = np.zeros(out_features)

        self.x: Optional[np.ndarray] = None
        self.x_shape = None

        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # テンソル対応
        self.x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

    def update_params(self, optimizer, layer_id):
        optimizer.update(
            params={f'W{layer_id}': self.W, f'b{layer_id}': self.b},
            grads={f'W{layer_id}': self.dW, f'b{layer_id}': self.db}
        )

    @property
    def info(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class Dense:
    """
    forward: x  ->  u  ->  out
               W,b     f
      x: input
      W: weight
      b: bias
      f: activation

      u: Affine, u = Wx + b
      h: output, h = f(u)

    backward:
      # todo
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: callable,
            init_method: callable = he_normal
    ):
        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            init_method=init_method
        )

        self.activation: Optional[callable] = activation()
        self.layer_id = None

        # for numerical grad
        self.init_W = self.linear.W.copy()
        self.init_b = self.linear.b.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        u = self.linear(x)

        if self.activation is None:
            return u
        return self.activation(u)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        if self.activation is None:
            dout = delta.copy()
        else:
            dout = self.activation.backward(delta)
        dout = self.linear.backward(dout)
        return dout

    def set_params(self, W: np.ndarray, b: np.ndarray):
        self.linear.W = W
        self.linear.b = b

    def set_id(self, id):
        self.layer_id = id

    @property
    def W(self) -> np.ndarray:
        return self.linear.W

    @property
    def b(self) -> np.ndarray:
        return self.linear.b

    @property
    def dW(self) -> np.ndarray:
        return self.linear.dW

    @property
    def db(self) -> np.ndarray:
        return self.linear.db

    def info(self):
        linear = self.linear.info
        activation = None if self.activation is None else self.activation.info
        return linear, activation

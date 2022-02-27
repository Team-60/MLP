"""
Adaline algo
"""

import numpy as np

np.random.seed(7)


class Perceptron:
    """
    Adaline perceptron
    """

    def __init__(
        self,
        dim: int,
        lr: int = 0.1,
        bias: bool = True,
        random_init: bool = True,
    ):
        self.dim = dim
        self.w = np.random.uniform(size=dim + 1) - 0.5 if random_init else np.zeros(dim + 1)
        self.bias = float(bias)
        self.lr = lr

    def forward(self, x: np.array) -> float:
        x_ = np.append(x, self.bias)
        self._affine_val = (self.w * x_).sum()
        return self._affine_val

    def threshold(self, v: float) -> int:
        return 1 if (v >= 0) else -1

    def predict(self, x: np.array) -> int:
        z = self.forward(x)
        return self.threshold(z)

    def update_weights(self, x: np.array, d: int):
        x_ = np.append(x, self.bias)
        z = self.forward(x)
        self.w += self.lr * (d - z) * x_

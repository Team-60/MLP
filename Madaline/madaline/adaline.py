"""
Adaline algo
"""

import numpy as np


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
        self.w = np.random.random(dim + 1) if random_init else np.zeros(dim + 1)
        self.bias = float(bias)
        self.lr = lr

    def forward(self, x: np.array) -> float:
        x_ = np.append(x, self.bias)
        return (self.w * x_).sum()

    def predict(self, x: np.array) -> int:
        y_pred = self.forward(x)
        return 1 if (y_pred >= 0) else -1

    def update_weights(self, x: np.array, d: int):
        x_ = np.append(x, self.bias)
        z = self.forward(x)
        self.w += self.lr * (d - z) * x_

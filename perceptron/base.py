"""
Base Perceptron Class
"""

import numpy as np


class Perceptron:
    """
    Rosenblatt's Perceptron
    """

    def __init__(self, dim: int, bias: bool = True, random_init: bool = True):
        self.dim = dim
        self.w = np.random.random(dim + 1) if random_init else np.zeros(dim + 1)
        self.bias = float(bias)

    def forward(self, x: np.array) -> float:
        x_ = np.append(x, self.bias)
        return (self.w * x_).sum()

    def predict(self, x: np.array, alt_class=False) -> int:
        y_pred = self.forward(x)
        neg_class = -1 if alt_class else 0
        return 1 if (y_pred >= 0) else neg_class

    def update_weights(self, x: np.array, dir: int):
        x_ = np.append(x, self.bias)
        delta_w = dir * x_
        self.w += delta_w

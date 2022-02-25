"""
Madaline algo
"""

import numpy as np
from . import adaline
from typing import Tuple

#### Layers ####


class Linear:
    """
    Linear layer in MLP
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float = 0.1,
        bias: bool = True,
        random_init: bool = True,
    ):
        self.bias = int(bias)
        self.perceptrons = [
            adaline.Perceptron(
                dim=input_size + 1,
                lr=lr,
                bias=bias,
                random_init=random_init,
            )
            for _ in range(output_size)
        ]

    def forward(self, x: np.array) -> np.array:
        output = np.array([p.predict(x) for p in self.perceptrons])
        return output


#### Model ####


class MLP:
    """
    Multilayer perceptron
    """

    def __init__(self, *layers: Tuple[Linear]):
        self.layers = list(layers)

    def forward(self, x: np.array) -> np.array:
        current_vector = x
        for layer in self.layers:
            current_vector = layer.forward(current_vector)
        return current_vector

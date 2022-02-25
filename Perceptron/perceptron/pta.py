"""
Perceptron Training Algorithm
"""

from . import base
import numpy as np
import copy

np.random.seed(7)


class WeightUpdateRepeat(Exception):
    def __init__(self, w):
        super().__init__(f"Repeated weights found! Non linearly seperable data exists. Weight vector {w}")


class PTA:
    """
    Perceptron Training Algorithm
    """

    def __init__(self, X: np.array, y: np.array, dim: int, bias: bool = True, random_init: bool = True, epsilon: float = 1e-5):
        self.perceptron = base.Perceptron(dim, bias, random_init)
        self.X = copy.deepcopy(X)
        self.y = np.array([i_ if i_ else -1 for i_ in y])
        self.epsilon = epsilon

        self._incorrect_sample = None
        self._previous_perceptron_weights = []

    def check_incorrect_sample(self) -> bool:
        for i in range(self.X.shape[0]):
            if self.perceptron.predict(self.X[i], alt_class=True) != self.y[i]:
                self._incorrect_sample = (self.X[i], self.y[i])
                return True
        return False

    def update_weights(self):
        sample_x, sample_y = self._incorrect_sample
        self._incorrect_sample = None
        self.perceptron.update_weights(sample_x, sample_y)

        current_w = np.copy(self.perceptron.w)
        self._check_repeating_weights(current_w)

    def _check_repeating_weights(self, new_w: np.array):
        for prev_w in self._previous_perceptron_weights:
            change_w = np.sqrt(((prev_w - new_w) ** 2).sum())
            if change_w < self.epsilon:
                raise WeightUpdateRepeat(new_w)

        self._previous_perceptron_weights.append(new_w)

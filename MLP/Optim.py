import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, x, x_grad):
        new_x = x - x_grad * self.learning_rate
        return new_x


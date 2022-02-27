import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, x, x_grad):
        new_x = x - x_grad * self.learning_rate
        return new_x

class SGD_momentum:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def optimize(self, x, x_grad, change):
        change = (1 - self.beta) * change - self.learning_rate * x_grad
        new_x = x + change
        return (new_x, change)

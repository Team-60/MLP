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

    def optimize(self, x, x_grad, momentum):
        momentum = self.beta * momentum - self.learning_rate * x_grad
        new_x = x + momentum
        return (new_x, momentum)

class NAG:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def optimize(self, x, x_delayed_grad, momentum):
        momentum = self.beta * momentum - self.learning_rate * x_delayed_grad
        new_x = x + momentum
        return (new_x, momentum)


class AdaGrad:
    def __init__(self, learning_rate, fudge_factor = 1e-6):
        self.learning_rate = learning_rate
        self.fudge_factor = fudge_factor

    def optimize(self, x, x_grad, g):
        new_x = x - x_grad * self.learning_rate / (np.sqrt(g + self.fudge_factor))
        return new_x

class RMSProp:
    def __init__(self, learning_rate, lamda, fudge_factor = 1e-6):
        self.learning_rate = learning_rate
        self.fudge_factor = fudge_factor
        self.lamda = lamda

    def optimize(self, x, x_grad, e):
        new_x = x - x_grad * self.learning_rate / (np.sqrt(e + self.fudge_factor))
        return new_x

class Adam:
    def __init__(self, learning_rate, lamda, beta, fudge_factor = 1e-6):
        self.learning_rate = learning_rate
        self.fudge_factor = fudge_factor
        self.lamda = lamda
        self.beta = beta

    def optimize(self, x, x_grad, momentum, e):
        momentum = self.beta * momentum - self.learning_rate * x_grad
        new_x = x + momentum * self.learning_rate / (np.sqrt(e + self.fudge_factor))
        return (new_x, momentum)

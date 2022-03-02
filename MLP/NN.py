import numpy as np
import ActivationFn
from copy import deepcopy
from Optim import SGD
from Linear import Linear
from ActivationLayer import ActivationLayer

class NN:

    def __init__(self):
        self.layers = []

    def __call__(self, x, no_grad=False):
        return self.forward(x, no_grad)

    def add_layers(self, *args):
        for arg in args:
            self.layers.append(arg)

    def add_optimizer(self, optimizer, opt_type):
        for layer in self.layers:
            layer.add_optimizer(optimizer, opt_type)

    def forward(self, x, no_grad=False):
        _x = deepcopy(x)
        for layer in self.layers:
            _x = layer.forward(_x, no_grad)
        return _x

    def backward(self, error):
        _error = deepcopy(error)
        for layer in reversed(self.layers):
            _error = layer.backward(_error)
        return _error

    def step(self):
        for layer in self.layers:
            layer.step()

if __name__ == "__main__":

    nn = NN()
    nn.add_layers(
        Linear(10, 20),
        ActivationLayer(ActivationFn.tanh, ActivationFn.tanh_derivative),
        Linear(20, 10),
        ActivationLayer(ActivationFn.tanh, ActivationFn.tanh_derivative),
        Linear(10, 1),
        ActivationLayer(ActivationFn.sigmoid, ActivationFn.sigmoid_derivative),
    )

    x = np.random.rand(10, 1)
    y = nn.forward(x)
    y_true = np.ones((1, 1))
    first_grad = nn.backward(2 * (y_true - y))
    sgd = SGD(0.01)
    nn.step(sgd)

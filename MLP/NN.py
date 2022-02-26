import numpy as np
import ActivationFn
from Optim import SGD
from Linear import Linear
from ActivationLayer import ActivationLayer

class NN:

    def __init__(self):
        self.layers = []

    def __call__(self, x):
        return self.forward(x)

    def add_layers(self, *args):
        for arg in args:
            self.layers.append(arg)

    def add_optimizer(optim, optim_derivative):
        self.optim = optim
        self.optim_derivative = optim_derivative

    def forward(self, x):
        _x = x.copy()
        for layer in self.layers:
            _x = layer.forward(_x)
        return _x

    def backward(self, error):
        _error = error.copy()
        for layer in reversed(self.layers):
            _error = layer.backward(_error)
        return _error

    def step(self, optimizer):
        for layer in self.layers:
            layer.step(optimizer)

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

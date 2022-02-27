
class ActivationLayer:

    def __init__(self, fn, fn_derivative):
        self.fn = fn
        self.fn_derivative = fn_derivative

    def forward(self, x, no_grad=False):
        return self.fn(x)

    def backward(self, x):
        return x * self.fn_derivative(x)

    def add_optimizer(self, optimizer, opt_type):
        pass

    def step(self):
        pass

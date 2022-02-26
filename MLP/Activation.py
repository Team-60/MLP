
class Activation:

    def __init__(self, fn, fn_derivative):
        self.fn = fn
        self.fn_derivative = fn_derivative

    def forward(self, x):
        return self.fn(x)

    def backward(self, x):
        return x * self.fn_derivative(x)

    def step(self, optimizer):
        pass

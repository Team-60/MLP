import numpy as np
import math

def sigmoid(x):
    return 1/(1 + math.e ** x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # TODO : leaky relu?
    return (x > 0) + 0


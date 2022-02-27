import numpy as np
import math
import time

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def softmax_derivative(x):
    return np.ones(x.shape)

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
    x = np.where(x < 0, 0, x)
    x = np.where(x >= 0, x, x)
    return x

def relu_derivative(x):
    # TODO : leaky relu?
    x = np.where(x < 0, 0, x)
    x = np.where(x >= 0, 1, x)
    return x

def softmax(x):
    exps = np.exp(x - x.max())
    #return exps / np.sum(exps, axis=0)
    #exps = np.exp(x)
    return exps / np.sum(exps, axis=0)

def softmax_derivative(x):
    return np.ones(x.shape)

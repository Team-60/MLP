import numpy as np
from time import sleep

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    # TODO : batch size
    return 2 * (y_pred - y_true)

 
def cross_entropy(y, output):
    '''
        L(y, ŷ) = −∑ylog(ŷ).
    '''
    l_sum = np.sum(np.multiply(y.T, np.log(output + 1e-15)))
    m = y.shape[0]
    l = -(1./m) * l_sum
    return l

def cross_entropy_derivative(y_true, y_pred):
    #print(y_pred * (1 - y_pred + 0.0001))
    #print(y_true)
    #sleep(1)
    return (y_pred - y_true)

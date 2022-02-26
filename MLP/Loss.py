import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    # TODO : batch size
    return 2 * (y_true - y_pred)

def BCE(y_true, y_pred):
    return np.sum(y_true * np.log(y_pred + 0.01) + (1 - y_true) * np.log(1 - y_pred - 0.01))

def BCE_derivative(y_true, y_pred):
    return (y_pred - y_true)

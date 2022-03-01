import numpy as np
import pandas as pd
import argparse

import ActivationFn
from train import train
from NN import NN
from Linear import Linear
import idx2numpy
from ActivationLayer import ActivationLayer

parser = argparse.ArgumentParser(description='Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=60000, help='batch size for training (default: 64)')
parser.add_argument('--num-epochs', type=int, default=500, help='number of epochs for training (default: 100)')
parser.add_argument('--momentum', type=int, default=0.9, help='momentum value of optimizer training (default: 0.9)')
parser.add_argument('--learning-rate', type=int, default=0.05, help='learning rate of optimizer training (default: 0.001)')
parser.add_argument('--display-interval', type=int, default=1000, help='interval for printing while training each epoch')
parser.add_argument('--experiment-name', type=str, default=None, help='Name of the experiment on model')
args = parser.parse_args()

def one_hot(y):
    y_new = np.zeros((y.size, y.max()+1))
    y_new[np.arange(y.size), y] = 1
    return y_new

def normalize(X):
    X = (X - X.mean())/X.std()
    return X

if __name__ == "__main__":

    np.random.seed(42)

    train_file = './Fashion-MNIST/train-images-idx3-ubyte'
    train_file_label = './Fashion-MNIST/train-labels-idx1-ubyte'
    X_train = idx2numpy.convert_from_file(train_file)
    y_train = idx2numpy.convert_from_file(train_file_label)
    test_file = './Fashion-MNIST/t10k-images-idx3-ubyte'
    test_file_label = './Fashion-MNIST/t10k-labels-idx1-ubyte'
    X_test = idx2numpy.convert_from_file(test_file)
    y_test = idx2numpy.convert_from_file(test_file_label)
    X_tr = np.zeros((len(X_train), 784))
    X_te = np.zeros((len(X_test), 784))
    for i in range(len(X_train)):
        X_tr[i] = np.ravel(X_train[i])
    for i in range(len(X_test)):
        X_te[i] = np.ravel(X_test[i])
    X_train = X_tr
    X_test = X_te

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    X_train = normalize(X_train)
    X_test = normalize(X_test)


    layers_config = [[784, 10], [784, 64, 10], [784, 256, 64, 10]]
    activations = ['relu', 'sigmoid', 'tanh']
    learning_rate = [1e-1, 0.05, 1e-2, 1e-3, 1e-4]

    for layers_shape in layers_config:
        for activation in activations:
            for lrate in learning_rate:

                print('===============================================')
                print('Config => Layers = {}, Activation = {}, Learnig_rate = {}'.format(layers_shape, activation, lrate))
                print("\n\n")

                args.learning_rate = lrate
                args.experiment_name = 'Layers-{}-Activation-{}-Rate-{}'.format(layers_shape, activation, lrate)

                layers = []
                for i in range(len(layers_shape) - 2):
                    if activation == 'relu':
                        actLayer = ActivationLayer(ActivationFn.relu, ActivationFn.relu_derivative)
                    elif activation == 'sigmoid':
                        actLayer = ActivationLayer(ActivationFn.sigmoid, ActivationFn.sigmoid_derivative)
                    elif activation == 'tanh':
                        actLayer = ActivationLayer(ActivationFn.tanh, ActivationFn.tanh_derivative)

                    layers.append(Linear(layers_shape[i], layers_shape[i + 1]))
                    layers.append(actLayer)

                le = len(layers_shape)
                layers.append(Linear(layers_shape[le - 2], layers_shape[le - 1]))
                actLayer = ActivationLayer(ActivationFn.softmax, ActivationFn.softmax_derivative)
                layers.append(actLayer)

                model = NN()
                model.add_layers(*layers)

                train(args, model, (X_train, y_train), (X_test, y_test))


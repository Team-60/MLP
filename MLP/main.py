import numpy as np
import pandas as pd
import argparse

import ActivationFn
from train import train
from NN import NN
from Linear import Linear
from ActivationLayer import ActivationLayer

parser = argparse.ArgumentParser(description='Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
parser.add_argument('--num-epochs', type=int, default=50, help='number of epochs for training (default: 100)')
parser.add_argument('--momentum', type=int, default=0.9, help='momentum value of optimizer training (default: 0.9)')
parser.add_argument('--learning-rate', type=int, default=0.1, help='learning rate of optimizer training (default: 0.001)')
parser.add_argument('--display-interval', type=int, default=5000, help='interval for printing while training each epoch')
args = parser.parse_args()

def split_label(df):

    label_dummies = pd.get_dummies(df['label'], prefix='label')
    df.pop('label')

    X = np.array(df)
    # TODO : std only on train
    X = (X - np.mean(X))/np.std(X)
    y = np.array(label_dummies)
    return X, y

if __name__ == "__main__":
    train_df = pd.read_csv('Fashion-MNIST/fashion-mnist_train.csv')
    test_df = pd.read_csv('Fashion-MNIST/fashion-mnist_train.csv')

    X_train, y_train = split_label(train_df)
    X_test, y_test = split_label(test_df)

    model = NN()
    model.add_layers(
        Linear(784, 10),
        ActivationLayer(ActivationFn.sigmoid, ActivationFn.sigmoid_derivative)
        #Linear(50, 10),
        #ActivationLayer(ActivationFn.sigmoid, ActivationFn.sigmoid_derivative),
    )

    train(args, model, (X_train, y_train))

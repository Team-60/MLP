import numpy as np
from NN import NN
import Loss
import Optim
import Loss
import matplotlib.pyplot as plt
import pickle
import os

def train(args, model, dataset, test_data):

    X_train, y_train = dataset

    optimizer = Optim.SGD(learning_rate=args.learning_rate)
    model.add_optimizer(optimizer, 'SGD')

    #optimizer = Optim.NAG(learning_rate=args.learning_rate, beta=0.9)
    #model.add_optimizer(optimizer, 'NAG')

    #optimizer = Optim.AdaGrad(learning_rate=args.learning_rate)
    #model.add_optimizer(optimizer, 'AdaGrad')

    #optimizer = Optim.RMSProp(learning_rate=args.learning_rate, lamda=0.9)
    #model.add_optimizer(optimizer, 'RMSProp')

    #optimizer = Optim.Adam(learning_rate=args.learning_rate, beta=0.9, lamda=0.9)
    #model.add_optimizer(optimizer, 'Adam')

    epochs = []
    losses = []
    train_accuracy = []
    test_accuracy = []


    for epoch in range(args.num_epochs):
        epoch_loss = 0

        permutation = np.random.permutation(X_train.shape[0])
        x_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for st_idx in range(0, len(X_train), args.batch_size):

            x_batch = x_train_shuffled[st_idx : min(st_idx + args.batch_size, len(x_train_shuffled))].T
            y_batch = y_train_shuffled[st_idx : min(st_idx + args.batch_size, len(y_train_shuffled))].T

            output = model(x_batch)

            loss = Loss.cross_entropy(y_batch, output) / args.batch_size
            loss_derivative = Loss.cross_entropy_derivative(y_batch, output)

            epoch_loss += loss

            model.backward(loss_derivative)
            model.step()

        outputs = np.argmax(model(x_train_shuffled.T, no_grad=True).T, axis=1)
        y_true = np.argmax(y_train_shuffled, axis=1)
        accuracy = np.sum(y_true == outputs)/args.batch_size

        num_batches = len(X_train) / args.batch_size
        test_acc = test(args, model, test_data)
        print('Epoch {} Finished with Loss : {}, Train Accuracy {:.5f}, Test Accuracy {:.5f}'.format(epoch, epoch_loss/num_batches, accuracy, test_acc))

        epochs.append(epoch)
        losses.append(epoch_loss/num_batches)
        train_accuracy.append(accuracy)
        test_accuracy.append(test_acc)

    os.makedirs('./models/{}'.format(args.experiment_name), exist_ok=True)
    with open('./models/{}/model.pickle'.format(args.experiment_name), 'wb')as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot(epochs, losses, train_accuracy, test_accuracy, args)


def plot(epochs, losses, train_accuracy, test_accuracy, args):
    plt.plot(epochs, train_accuracy, label='train')
    plt.plot(epochs, test_accuracy, label='test')
    plt.legend()
    plt.savefig('./models/{}/plot-accuracy.png'.format(args.experiment_name))
    plt.figure()

    plt.plot(epochs, losses, label='loss')
    plt.legend()
    plt.savefig('./models/{}/plot-loss.png'.format(args.experiment_name))

def test(args, model, dataset):

    X_test, y_test = dataset

    outputs = np.argmax(model(X_test.T, no_grad=True).T, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.sum(y_true == outputs)/X_test.shape[0]

    return accuracy


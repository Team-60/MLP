import numpy as np
from NN import NN
import Loss
import Optim
import Loss
import matplotlib.pyplot as plt

def train(args, model, dataset, test_data):

    np.random.seed(42)
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

            batch_loss = 0
            batch_loss_derivative = 0
            for idx in range(st_idx, min(len(X_train), st_idx + args.batch_size)):

                labels = y_train_shuffled[idx]
                image = x_train_shuffled[idx]

                image = image.reshape(784, 1)
                labels = labels.reshape(10, 1)

                output = model(image)

                loss = Loss.cross_entropy(labels, output)
                batch_loss += loss
            
                loss_derivative = Loss.cross_entropy_derivative(labels, output)
                batch_loss_derivative += loss_derivative

            batch_loss /= args.batch_size
            batch_loss_derivative /= args.batch_size

            epoch_loss += batch_loss

            model.backward(batch_loss_derivative)
            model.step()

            #if idx / args.batch_size % args.display_interval == 0:
            #    print('Train Epoch: {} [{}/{}]\tLoss: {:0.6f}'.format(
            #        epoch, idx, len(X_train), loss))

        correct = 0
        total = 0
        for idx, image in enumerate(X_train):

            labels = y_train[idx]
            image = image.reshape(784, 1)
            labels = labels.reshape(10, 1)
            output = model(image, no_grad=True)

            prediction = np.argmax(output)
            true = np.argmax(labels)

            correct += (prediction == true)
            total += 1

        num_batches = len(X_train) / args.batch_size
        test_acc = test(args, model, test_data)
        print('Epoch {} Finished with Loss : {}, Accuracy {:.2f}'.format(epoch, epoch_loss/num_batches, correct/total))

        epochs.append(epoch)
        losses.append(epoch_loss/num_batches)
        train_accuracy.append(correct/total)
        test_accuracy.append(test_acc)

    plot(epochs, epoch_loss, train_accuracy, test_accuracy)

def plot(epochs, epoch_loss, train_accuracy, test_accuracy):
    plt.plot(epochs, train_accuracy, label='train')
    plt.plot(epochs, test_accuracy, label='test')
    plt.legend()
    plt.show()


def test(args, model, dataset):

    X_test, y_test = dataset
    correct = 0
    total = 0
    for idx, image in enumerate(X_test):

        labels = y_test[idx]
        image = image.reshape(784, 1)
        labels = labels.reshape(10, 1)
        output = model(image, no_grad=True)

        prediction = np.argmax(output)
        true = np.argmax(labels)

        correct += (prediction == true)
        total += 1

    print('Finished Test Accuracy {:.2f}'.format(correct/total))
    return correct/total


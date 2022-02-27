import numpy as np
from NN import NN
import Loss
import Optim
import Loss

def train(args, model, dataset):

    np.random.seed(42)
    X_train, y_train = dataset

    optimizer = Optim.SGD(learning_rate=args.learning_rate)

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for idx, image in enumerate(X_train):

            labels = y_train[idx]
            image = image.reshape(784, 1)
            labels = labels.reshape(10, 1)
            output = model(image)

            loss = Loss.cross_entropy(labels, output)
            epoch_loss += loss

            loss_derivative = Loss.cross_entropy_derivative(labels, output)

            model.backward(loss_derivative)
            model.step(optimizer)

            if idx % args.display_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:0.6f}'.format(
                    epoch, idx, len(X_train), loss))

        correct = 0
        total = 0
        for idx, image in enumerate(X_train):

            image = image.reshape(784, 1)
            labels = labels.reshape(10, 1)
            output = model(image)
            labels = y_train[idx]

            prediction = np.argmax(output)
            true = np.argmax(labels)

            correct += (prediction == true)
            total += 1

        print('Epoch {} Finished with Total Loss : {}, Accuracy {:.2f}'.format(epoch, epoch_loss, correct/total))


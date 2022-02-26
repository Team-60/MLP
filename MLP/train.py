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

            loss = Loss.mse(output, labels)
            epoch_loss += loss

            loss_derivative = Loss.mse_derivative(output, labels)

            model.backward(loss_derivative)
            model.step(optimizer)

            if idx % args.display_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:0.6f}'.format(
                    epoch, idx, len(X_train), loss))

        print('Epoch {} Finished with Total Loss : {}'.format(epoch, epoch_loss))


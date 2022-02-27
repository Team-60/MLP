import numpy as np
from tqdm import tqdm
import sys
import pickle


exp_name = "default"
if len(sys.argv) > 1:
    exp_name = sys.argv[1]


############### ADALINE CHECK ###############

# from madaline.adaline import Perceptron

# # INPUTS
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 0, 0, 1])

# y = np.array([1 if it else -1 for it in y])

# percep = Perceptron(2, random_init=False)


# def get_incorrect():
#     for i in range(len(X)):
#         if percep.predict(X[i]) != y[i]:
#             return X[i], y[i]
#     return None


# iters = 0
# while sample := get_incorrect():
#     sx, sy = sample
#     percep.update_weights(sx, sy)
#     iters += 1
#     if iters > 1000:
#         raise Exception

# print("ITERS", iters)
# print("WEIGHTS", percep.w)
# for i in range(X.shape[0]):
#     print()
#     print("Sample", X[i])
#     print("Perceptron output", percep.predict(X[i]))
#     print("Actual output", y[i])


############### MADALINE CHECK ###############

import matplotlib.pyplot as plt


def scale(X):
    X[:, 0] /= X[:, 0].max()
    X[:, 1] /= X[:, 1].max()
    X[:, 0] -= X[:, 0].mean()
    X[:, 1] -= X[:, 1].mean()
    return X


def plot_model(model):
    my = np.array([model.forward(sx) for sx in X])
    pX = X[my == 1]
    plt.scatter(pX[:, 0], pX[:, 1], s=0.4)
    nX = X[my == -1]
    plt.scatter(nX[:, 0], nX[:, 1], s=0.4)
    plt.show()


# INPUTS
def gen_dataset():
    def get_rect(x1: int, y1: int, x2: int, y2: int, num: int) -> np.array:
        x1 += 0.5
        y1 += 0.5

        x2 -= 0.5
        y2 -= 0.5

        num_ = int(np.sqrt(num))
        assert num_ ** 2 == num
        return [[px, py] for px in np.linspace(x1, x2, num_) for py in np.linspace(y1, y2, num_)]

    X = []
    y = []

    # positive samples
    pos_num_points = 49

    X += get_rect(0, 4, 2, 6, pos_num_points)
    y += [1] * pos_num_points

    X += get_rect(4, 0, 6, 2, pos_num_points)
    y += [1] * pos_num_points

    X += get_rect(8, 4, 10, 6, pos_num_points)
    y += [1] * pos_num_points

    X += get_rect(4, 8, 6, 10, pos_num_points)
    y += [1] * pos_num_points

    X += get_rect(4, 4, 6, 6, pos_num_points)
    y += [1] * pos_num_points

    # negative samples
    neg_num_points = 25

    X += get_rect(2, 4, 4, 6, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(4, 2, 6, 4, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(6, 4, 8, 6, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(4, 6, 6, 8, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(0, 6, 4, 10, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(0, 0, 4, 4, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(6, 0, 10, 4, neg_num_points)
    y += [0] * neg_num_points

    X += get_rect(6, 6, 10, 10, neg_num_points)
    y += [0] * neg_num_points

    # plot
    X, y = scale(np.array(X)), np.array(y)
    pX = X[y == 1]
    plt.scatter(pX[:, 0], pX[:, 1], s=0.4)
    nX = X[y == 0]
    plt.scatter(nX[:, 0], nX[:, 1], s=0.4)
    plt.show()

    return X, y


X, y = gen_dataset()

y = np.array([1 if it else -1 for it in y])

from madaline.madaline import MLP, Linear

# Load model
with open(f"weights/{exp_name}.pkl", "rb") as fp:
    model = pickle.load(fp)
    print("Loaded", exp_name, "; accuracy:", model.get_acc(X, y))
    plot_model(model)
    exit(1)

model = MLP(
    Linear(2, 64),
    Linear(64, 32),
    Linear(32, 16),
    Linear(16, 8),
    Linear(8, 1),
)
# model._debug = True

EPOCHS = 10
mv = 0
for epoch in (pbar := tqdm(range(EPOCHS))):
    updated = False
    for i in range(len(X)):
        sx, sy = X[i], y[i]
        y_p = model.forward(sx)
        if y_p != sy:
            updated = True
            model.update_weights(sx, sy, X, y)
            current_acc = model.get_acc(X, y)
            mv = max(mv, current_acc)
            pbar.set_description(f"Epoch {epoch}, Acc {current_acc}, Max {mv}")
    if not updated:
        break

    plot_model(model)

# Save model
with open(f"weights/{exp_name}.pkl", "wb") as fp:
    pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)

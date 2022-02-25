import numpy as np

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

# # INPUTS
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

from madaline.madaline import MLP, Linear

model = MLP(
    Linear(2, 2),
    Linear(2, 1)
)

print(model.forward(X[3]))
import numpy as np
from perceptron.pta import PTA
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"

# INPUTS
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])


def save_plot(iter, w):
    plt.figure()

    positives = X[y == 1]
    for s in positives:
        plt.scatter(s[0], s[1], color="g")
    negatives = X[y == 0]
    for s in negatives:
        plt.scatter(s[0], s[1], color="r")

    lin_x = np.linspace(-0.01, 1.01, 100)
    if w[1] == 0:
        lin_y = np.zeros_like(lin_x)
    else:
        lin_y = -1 * (w[0] * lin_x + w[2]) / w[1]

    plt.plot(lin_x, lin_y, color="b")
    plt.title(f"Decision boundary at iter={iter}")
    plt.savefig(f"{OUTPUT_DIR}/{iter}.png")


pta_algo = PTA(X, y, 2, random_init=False)

iters = 0
while pta_algo.check_incorrect_sample():
    save_plot(iters, pta_algo.perceptron.w)
    try:
        pta_algo.update_weights()
    except:
        print("CURRENT STEP", iters)
        raise
    iters += 1

save_plot(iters, pta_algo.perceptron.w)

print(f"* Convergence in {iters} iters")
print(f"* Final weights: {pta_algo.perceptron.w}")

for i in range(X.shape[0]):
    print()
    print("Sample", X[i])
    print("Perceptron output", pta_algo.perceptron.predict(X[i]))
    print("Actual output", y[i])

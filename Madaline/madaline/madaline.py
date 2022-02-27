"""
Madaline algo
"""

from copy import deepcopy
import numpy as np
from typing import Tuple

from . import adaline

#### Layers ####


class Linear:
    """
    Linear layer in MLP
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float = 0.001,
        bias: bool = True,
        random_init: bool = True,
    ):
        self.bias = int(bias)
        self.input_size = input_size
        self.output_size = output_size
        self.perceptrons = [
            adaline.Perceptron(
                dim=input_size,
                lr=lr,
                bias=bias,
                random_init=random_init,
            )
            for _ in range(output_size)
        ]
        self.num_p = len(self.perceptrons)

        if random_init:  # perform xavier initialisation
            xavier_term = np.sqrt(6 / (self.input_size + self.output_size))
            for i in range(self.num_p):
                percep = self.perceptrons[i]
                percep.w = np.random.uniform(
                    low=-1 * xavier_term,
                    high=xavier_term,
                    size=self.input_size + 1,
                )

    def forward(self, x: np.array) -> np.array:
        output = np.array([p.predict(x) for p in self.perceptrons])
        return output


#### Model ####


class MLP:
    """
    Multilayer perceptron
    """

    def __init__(self, *layers: Tuple[Linear]):
        self.layers = layers
        self.num_l = len(self.layers)
        self._debug = False
        assert self.layers[-1].output_size == 1, "Last layer output should be single valued!"

    def get_acc(self, X: np.array, y: np.array) -> float:
        correct = 0
        for i in range(len(X)):
            correct += self.forward(X[i]) == y[i]
        return correct / len(X)

    def forward(self, x: np.array) -> np.array:
        current_vector = x
        for layer in self.layers:
            current_vector = layer.forward(current_vector)
        return current_vector.item()

    def update_weights(self, x: np.array, y: int, dX: np.array, dy: np.array):
        assert self.forward(x) != y, "Correct output already exists!"

        # find neuron on based on min abs affine values
        neurons = [(l_i, p_i) for l_i in range(self.num_l) for p_i in range(self.layers[l_i].num_p)]
        affine_sorted = sorted(neurons, key=lambda id: (id[0], abs(self.layers[id[0]].perceptrons[id[1]]._affine_val)))

        if self._debug:
            print("fix", x, y)
            print("id", neurons)
            print("weights", [self.layers[l_i].perceptrons[p_i].w for l_i in range(self.num_l) for p_i in range(self.layers[l_i].num_p)])
            print("affine val", [self.layers[l_i].perceptrons[p_i]._affine_val for l_i in range(self.num_l) for p_i in range(self.layers[l_i].num_p)])
            print("affine sorted", affine_sorted)

        current_acc = self.get_acc(dX, dy)

        def adaline_flip_ok(cl_i: int, cp_i: int, d: int):
            if cl_i == 0:
                input_v = x
            else:
                input_v = np.array([p.threshold(p._affine_val) for p in self.layers[cl_i - 1].perceptrons])

            fix_neuron = self.layers[cl_i].perceptrons[cp_i]
            prev_fix_neuron_w = deepcopy(fix_neuron.w)
            while fix_neuron.predict(input_v) != d:
                fix_neuron.update_weights(input_v, d)

            if self._debug:
                print("min neuron", cl_i, cp_i)
                print("input_v", input_v)
                print("fix neuron w", fix_neuron.w)
                print()

            # get new acc
            new_acc = self.get_acc(dX, dy)
            if new_acc >= current_acc:
                return True
            else:
                fix_neuron.w = prev_fix_neuron_w
                return False

        for cl_i, cp_i in affine_sorted:

            # reform _affine_val
            self.forward(x)

            # check wether fixing neuron, changes the output
            l_i_outv = np.array([p.threshold(p._affine_val) for p in self.layers[cl_i].perceptrons])
            l_i_outv[cp_i] = -1 if (l_i_outv[cp_i] == 1) else 1  # flip the neuron output

            if self._debug:
                print("> checking", cl_i, cp_i)
                print(f"> modified out at {cl_i}", l_i_outv)

            # find the output with the modified vector
            current_vector = deepcopy(l_i_outv)
            for layer_ in self.layers[cl_i + 1 :]:
                current_vector = layer_.forward(current_vector)

            # fix neuron
            if current_vector.item() == y:
                if adaline_flip_ok(cl_i, cp_i, l_i_outv[cp_i]):
                    return

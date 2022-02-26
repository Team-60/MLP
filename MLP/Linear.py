import numpy as np
import time

class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # TODO : random initialization
        self.W = np.random.rand(self.in_features, self.out_features)
        self.b = np.random.rand(out_features, 1)

    def forward(self, layer_input):
        # shapes : (in_features, 1)  -> (out_features, 1)
        self.last_input = layer_input
        output = (self.W.T @ layer_input) + self.b
        #print(layer_input)
        return output
    
    def backward(self, nextl_gradients):
        # shapes : (out_features, 1) -> (input_features, 1)
        self.weight_grads = self.last_input @ nextl_gradients.T
        self.bias_grads = nextl_gradients.copy()

        gradients_to_prop = self.W @ nextl_gradients
        return gradients_to_prop

    def step(self, optimizer):
        self.W = optimizer.optimize(self.W, self.weight_grads)
        self.b = optimizer.optimize(self.b, self.bias_grads)
        

if __name__ == "__main__":

    layer = Linear(5, 10)
    layer.forward(np.zeros(5))



import numpy as np
from copy import deepcopy

class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.last_input = 0
        self.count = 0

        self.W = np.random.rand(self.in_features, self.out_features)
        self.b = np.random.rand(out_features, 1)

    
    def add_optimizer(self, optimizer, opt_type):

        self.opt_type = opt_type
        self.optimizer = optimizer

        if opt_type == 'SGD':
            pass
        elif opt_type == 'SGD_momentum':
            self.momentum_W = 0
            self.momentum_b = 0
        elif self.opt_type == 'NAG':
            self.d_last_input = 0
            self.momentum_W = 0
            self.momentum_b = 0
        elif self.opt_type == 'AdaGrad':
            self.G_W = 0
            self.G_d = 0
        elif self.opt_type == 'RMSProp':
            self.E_W = 0
            self.E_b = 0
        elif self.opt_type == 'Adam':
            self.momentum_W = 0
            self.momentum_b = 0
            self.E_W = 0
            self.E_b = 0
        else:
            print("Error : No such optimizer defined")


    def forward(self, layer_input, no_grad=False):
        # shapes : (in_features, 1)  -> (out_features, 1)
        if no_grad == False:
            self.last_input += layer_input
            self.count += 1

        if self.opt_type == 'NAG' and no_grad == False:
            self.d_W = self.W + self.momentum_W
            self.d_b = self.b + self.momentum_b
            self.d_output = (self.d_W.T @ layer_input) + self.d_b
            return self.d_output

        output = (self.W.T @ layer_input) + self.b
        return output
    
    def backward(self, nextl_gradients):
        # shapes : (out_features, 1) -> (input_features, 1)
        self.weight_grads = self.last_input @ nextl_gradients.T
        self.bias_grads = deepcopy(nextl_gradients)

        self.weight_grads /= self.count
        self.bias_grads /= self.count

        gradients_to_prop = self.W @ nextl_gradients

        self.last_input = 0
        self.count = 0

        if self.opt_type == 'AdaGrad':
            self.G_W += self.weight_grads ** 2 
            self.G_d += self.bias_grads ** 2
        elif self.opt_type == 'RMSProp':
            self.E_W = self.E_W * self.optimizer.lamda + (1 - self.optimizer.lamda) * (self.weight_grads ** 2)
            self.E_b = self.E_b * self.optimizer.lamda + (1 - self.optimizer.lamda) * (self.bias_grads ** 2)
        elif self.opt_type == 'Adam':
            self.E_W = self.E_W * self.optimizer.lamda + (1 - self.optimizer.lamda) * (self.weight_grads ** 2)
            self.E_b = self.E_b * self.optimizer.lamda + (1 - self.optimizer.lamda) * (self.bias_grads ** 2)
        
        return gradients_to_prop

    def step(self):

        if self.opt_type == 'SGD':
            self.W = self.optimizer.optimize(self.W, self.weight_grads)
            self.b = self.optimizer.optimize(self.b, self.bias_grads)
        elif self.opt_type == 'SGD_momentum':
            self.W, self.momentum_W = self.optimizer.optimize(self.W, self.weight_grads, self.momentum_W)
            self.b, self.momentum_b = self.optimizer.optimize(self.b, self.bias_grads, self.momentum_b)
        elif self.opt_type == 'NAG':
            self.W, self.momentum_W = self.optimizer.optimize(self.W, self.weight_grads, self.momentum_W)
            self.b, self.momentum_b = self.optimizer.optimize(self.b, self.bias_grads, self.momentum_b)
        elif self.opt_type == 'AdaGrad':
            self.W = self.optimizer.optimize(self.W, self.weight_grads, self.G_W)
            self.b = self.optimizer.optimize(self.b, self.bias_grads, self.G_d)
        elif self.opt_type == 'RMSProp':
            self.W = self.optimizer.optimize(self.W, self.weight_grads, self.E_W)
            self.b = self.optimizer.optimize(self.b, self.bias_grads, self.E_b)
        elif self.opt_type == 'Adam':
            self.W, self.momentum_W = self.optimizer.optimize(self.W, self.weight_grads, self.momentum_W, self.E_W)
            self.b, self.momentum_b = self.optimizer.optimize(self.b, self.bias_grads, self.momentum_b, self.E_b)
        else:
            print("Layer.step(), Invalid Optimizer")
            exit(0)

if __name__ == "__main__":

    layer = Linear(5, 10)
    layer.forward(np.zeros(5))



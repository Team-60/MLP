import numpy as np
from copy import deepcopy

class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.last_input = 0
        self.count = 0

        xavier_term = np.sqrt(6 / (self.in_features + self.out_features))
        self.W = np.random.uniform(-xavier_term, xavier_term, size=(in_features, out_features))
        self.b = np.random.uniform(-xavier_term, xavier_term, size=(out_features, 1))
        #self.W = np.random.rand(self.in_features, self.out_features)
        #self.b = np.random.rand(out_features, 1)

    
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
        # shapes : (in_features, batch_size)  -> (out_features, batch_size)
        if no_grad == False:
            self.last_input = layer_input

        if (self.opt_type == 'NAG') and (no_grad == False):
            self.d_W = self.W + self.momentum_W
            self.d_b = self.b + self.momentum_b
            self.d_output = (self.d_W.T @ layer_input) + self.d_b
            return self.d_output

        output = (self.W.T @ layer_input) + self.b
        return output
    
    def backward(self, nextl_gradients):
        # shapes : (out_features, batch_size) -> (input_features, 1)
        # last_input = (in_features, batch_size)
        # self.weight_grads = (in_features, out_features)
        self.weight_grads = (self.last_input @ nextl_gradients.T) / nextl_gradients.shape[1]
        self.bias_grads = np.mean(nextl_gradients, axis=1).reshape(-1, 1)

        assert(self.weight_grads.shape == (self.in_features, self.out_features))
        assert(self.bias_grads.shape == (self.out_features, 1))

        gradients_to_prop = self.W @ nextl_gradients
        # gradients_to_prop = (in_features batch_sizes)

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



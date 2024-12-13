
import numpy as np
#from .Optimizers import Optimizer
import common.Optimizers as Optimizers

class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, optimizer):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.W, self.B = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        self.opt_W = Optimizers.Optimizer.create_optimizer(optimizer)
        self.opt_B = Optimizers.Optimizer.create_optimizer(optimizer)

    def Update(self, lr):
        self.W = self.opt_W.update(lr, self.W, self.dW)
        self.B = self.opt_B.update(lr, self.B, self.dB)

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == "zero":
            W = np.zeros((num_input, num_output))
        elif method == "normal":
            W = np.random.normal(size=(num_input, num_output))
        elif method == "kaiming":
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output))
        elif method == "xavier":
            # xavier
            W = np.random.uniform(
                -np.sqrt(6/(num_output + num_input)),
                np.sqrt(6/(num_output + num_input)),
                size=(num_input, num_output)
            )
        else:
            raise Exception("Unknown init method")
        B = np.zeros((1, num_output))
        return W, B

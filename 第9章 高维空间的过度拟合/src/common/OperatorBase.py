import numpy as np
import os
import sys

# 把所有的操作都看作是 Operator, 相当于PyTorch中的Layer
class Operator(object):
    def __init__(self):
        pass

    def forward(self, z):
        pass

    def backward(self, z, a):
        pass

    def update(self):
        pass

    def load(self, name):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name)
        return np.loadtxt(file_path)

    def save(self, name, value):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name)
        np.savetxt(file_path, value, fmt="%f")


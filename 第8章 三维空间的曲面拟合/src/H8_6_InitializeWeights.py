import os
import sys
import math
import numpy as np
import common.Layers_6 as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_8 import DataLoader_8
from common.TrainingHistory_8 import TrainingHistory_8
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import r2
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


# 自定义的模型
class NN(Module):
    def __init__(self, num_hidden, activator, init_method="normal"):
        self.linear1 = layer.Linear(num_hidden, num_hidden, init_method)
        self.activator1 = activator
        self.linear2 = layer.Linear(num_hidden, num_hidden, init_method)
        self.activator2 = activator
        self.linear3 = layer.Linear(num_hidden, num_hidden, init_method)
        self.activator3 = activator
        self.linear4 = layer.Linear(num_hidden, num_hidden, init_method)
        self.activator4 = activator
        self.linear5 = layer.Linear(num_hidden, 1, init_method)

    def forward(self, X):
        Z1 = self.linear1.forward(X)
        A1 = self.activator1.forward(Z1)
        Z2 = self.linear2.forward(A1)
        A2 = self.activator2.forward(Z2)
        Z3 = self.linear3.forward(A2)
        A3 = self.activator3.forward(Z3)
        Z4 = self.linear4.forward(A3)
        A4 = self.activator4.forward(Z4)
        self.predict = self.linear5.forward(A4)
        return A1, A2, A3, A4
    
    # X:输入批量样本, Y:标签, Z:预测值
    def backward(self, label):
        dZ5 = self.predict - label
        dA4 = self.linear5.backward(dZ5)
        dZ4 = self.activator4.backward(dA4)
        dA3 = self.linear4.backward(dZ4)
        dZ3 = self.activator3.backward(dA3)
        dA2 = self.linear3.backward(dZ3)
        dZ2 = self.activator2.backward(dA2)
        dA1 = self.linear2.backward(dZ2)
        dZ1 = self.activator1.backward(dA1)
        self.linear1.backward(dZ1)
        return dA1, dA2, dA3, dA4

def show_forward_hist(A):
    a_value = []
    for a in A:
        a_value.append(a)

    for i in range(len(a_value)):
        ax = plt.subplot(1, 4, i+1)
        ax.set_title("layer" + str(i+1))
        plt.ylim(0,5000)
        if i > 0:
            plt.yticks([])
        ax.hist(a_value[i].flatten(), bins=25, range=[-1, 1])
    #end for
    plt.show()

def show_backward_hist(A):
    count = 100
    x = np.linspace(-0.2, 0.2, count + 1)
    counter = np.zeros(count)
    for i, a in enumerate(A):
        values = a.flatten()
        ax = plt.subplot(1, 4, i+1)
        ax.hist(values, bins=25, range=[-2, 2], density=True)
        # print(np.max(values), np.min(values))
        # for i in range(count):
        #     b = (values >= x[i]) & (values < x[i+1])
        #     counter[i] = b.sum()
        # print(counter)
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    num_hidden = 64
    num_sample = 1000
    x = np.random.randn(num_sample, num_hidden)
    model = NN(num_hidden, activator.Tanh(), init_method="normal")
    A = model.forward(x)
    show_forward_hist(A)
    # dA = model.backward(np.random.normal(0, 1, (num_sample, 1)))
    # show_backward_hist(dA)
    model = NN(num_hidden, activator.Tanh(), init_method="xavier")
    A = model.forward(x)
    show_forward_hist(A)

    model = NN(num_hidden, activator.Relu(), init_method="xavier")
    A = model.forward(x)
    show_forward_hist(A)

    model = NN(num_hidden, activator.Relu(), init_method="kaiming")
    A = model.forward(x)
    show_forward_hist(A)

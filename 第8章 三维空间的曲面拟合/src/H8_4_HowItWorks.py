import os
import sys
import math
import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_8 import DataLoader_8
from common.TrainingHistory_8 import TrainingHistory_8
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import r2
import matplotlib.pyplot as plt
from H8_3_Train import *


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


# 自定义的模型
class NN(Module):
    def __init__(self):
        # 初始化 forward 中需要的各自operator
        self.linear1 = layer.Linear(2, 12)
        self.tanh = activator.Tanh()
        self.linear2 = layer.Linear(12, 1)
        self.loss = loss.MSE()

    def forward(self, X):
        self.Z1 = self.linear1.forward(X)
        self.A1 = self.tanh.forward(self.Z1)
        Z2 = self.linear2.forward(self.A1)
        return Z2

    def predict(self, X):
        return self.forward(X)

    def load(self, name):
        super().load_parameters(name, (self.linear1, self.linear2))

def show(model):
    count = 100
    x = np.linspace(-4, 4, count)
    y = np.linspace(-4, 4, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    A = model.predict(input)
    # 隐层线性输出斜平面
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = model.Z1[:,idx].reshape(count, count)
            ax.plot_surface(X,Y,Z,cmap='rainbow')
            title = str.format("${0:.2f}x_1+{1:.2f}x_2+{2:.2f}$", 
                               model.linear1.W[0, idx], 
                               model.linear1.W[1, idx], 
                               model.linear1.B[0, idx])
            ax.set_title(title)
    plt.show()
    # 激活输出
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = model.A1[:,idx].reshape(count, count)
            ax.plot_surface(X,Y,Z,cmap='rainbow')
            title = str.format("$w={0:.2f}$", model.linear2.W[idx, 0])
            ax.set_title(title)
    plt.show()
    # 分步叠加
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = np.zeros((idx+1, count, count))
            for k in range(idx+1):
                Z[k] = model.A1[:,k].reshape(count, count) * model.linear2.W[k,0]
            ax.plot_surface(X, Y, np.sum(Z, axis=0) ,cmap='rainbow')
            ax.set_title(str(idx+1))
    plt.show()



if __name__=="__main__":
    data_loader = load_data()
    params = HyperParameters(max_epoch=5000, batch_size=32, learning_rate=0.01)
    model = NN()
    model.load("my_model")
    show(model)




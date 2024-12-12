import os
import sys
import math
import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_10 import DataLoader_10
from common.Module import Module
import matplotlib.pyplot as plt
from H10_3_NN_train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


if __name__=="__main__":
    data_loader = load_data()
    model = NN()
    model.load("model_10_3")
    X, Y = data_loader.get_val()
    X0 = X[Y==0][:, np.newaxis]
    X1 = X[Y==1][:, np.newaxis]
    Z10 = model.linear1.forward(X0) # 1D -> 2D
    A10 = model.tanh.forward(Z10)   # 2D linear -> 2D curve
    Z20 = model.linear2.forward(A10)
    Z11 = model.linear1.forward(X1) # 1D -> 2D
    A11 = model.tanh.forward(Z11)   # 2D linear -> 2D curve
    Z21 = model.linear2.forward(A11)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(X0, [0]*X0.shape[0], marker='.', label="0")
    ax.scatter(X1, [0]*X1.shape[0], marker='x', label="1")
    ax.grid()
    ax.legend()
    ax.set_title("1.原始数据(归一化)")
    ax.set_xlabel("$x$")
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(Z10[:,0], Z10[:,1], marker='.', label="0")
    ax.scatter(Z11[:,0], Z11[:,1],  marker='x', label="1")
    ax.grid()
    ax.legend()
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("2.第一层线性输出(增加一维特征)")
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(A10[:,0], A10[:,1], marker='.', label="0")
    ax.scatter(A11[:,0], A11[:,1],  marker='x', label="1")
    ax.grid()
    ax.legend()
    ax.set_xlabel("$a_1$")
    ax.set_ylabel("$a_2$")
    ax.set_title("3.第一层激活输出(弯曲)")
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(Z20, [0]*Z20.shape[0], marker='.', label="0")
    ax.scatter(Z21, [0]*Z21.shape[0],  marker='x', label="1")
    ax.grid()
    ax.legend()
    ax.set_title("4.第二层线性输出(折叠)")
    ax.set_xlabel("$z$")
    plt.show()
    
    print("----计算实例----")
    x = np.array([[0.4321]])
    z1 = model.linear1.forward(x)
    a1 = model.tanh.forward(z1)
    z2 = model.linear2.forward(a1)
    print("输入值：", x)
    print("第一层输出：", z1)
    print("第一层激活输出：", a1)
    print("第二层输出：", z2)
    print("预测值：", model.predict(x))

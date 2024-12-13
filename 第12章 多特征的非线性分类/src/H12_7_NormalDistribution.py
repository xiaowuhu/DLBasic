
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

import common.Activators as activator

def nd_fun(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f


def func_sigmoid(X, W):
    Z = np.dot(X, W)
    A = activator.Sigmoid().forward(Z)
    return A

def func_tanh(X, W):
    Z = np.dot(X, W)
    A = activator.Tanh().forward(Z)
    return A

def func_relu(X, W):
    Z = np.dot(X, W)
    A = activator.Relu().forward(Z)
    return A

def show_tanh_sigmoid_relu():
    X = np.random.normal(loc=0, scale=1, size=(100,8))
    W = np.random.randn(8, 10)
    A_tanh = func_tanh(X, W)
    A_tanh_1 = func_tanh(X, W+1)
    A_sigmoid = func_sigmoid(X, W)
    A_relu = func_relu(X, W)
    
    x = np.linspace(-5,5)
    fX = nd_fun(x, np.std(X), np.mean(X))
    fig = plt.figure(figsize=(8,8))

    fA_tanh = nd_fun(x, np.std(A_tanh), np.mean(A_tanh))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, fX, label='data X')
    ax.plot(x, fA_tanh, label='$Tanh(X \cdot W)$', linestyle=':')
    ax.grid()
    ax.legend()

    fA_tanh1 = nd_fun(x, np.std(A_tanh_1), np.mean(A_tanh_1))
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, fX, label='data X')
    ax.plot(x, fA_tanh1, label='$Tanh(X \cdot (W+1))$', linestyle=':')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    fA_sigmoid = nd_fun(x, np.std(A_sigmoid), np.mean(A_sigmoid))   
    ax.plot(x, fX, label='data X')
    ax.plot(x, fA_sigmoid, label='$Sigmoid(X \cdot W)$', linestyle=':')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    fA_relu = nd_fun(x, np.std(A_relu), np.mean(A_relu))   
    ax.plot(x, fX, label='data X')
    ax.plot(x, fA_relu, label='$ReLU(X \cdot W)$', linestyle=':')
    ax.grid()
    ax.legend()

    plt.show()


def show_3_sigmoid():
    
    x = np.linspace(-5,5)
    y = activator.Sigmoid().forward(x)
    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(x, y, linestyle=":")
    ax.grid()
    ax.set_title(r"原始数据分布 $x$")

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(x, y, linestyle=":")
    ax.grid()
    ax.set_title(r"$y=(x - \mu)/\sigma$")

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(x, y, linestyle=":")
    ax.grid()
    ax.set_title(r"$z=\gamma · y + \beta$")

    plt.show()


if __name__ == '__main__':

    show_tanh_sigmoid_relu()

    show_3_sigmoid()








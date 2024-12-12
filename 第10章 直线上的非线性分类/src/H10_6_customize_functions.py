import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from common.DataLoader_10 import DataLoader_10

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def bce_loss_function(a,y):
    p1 = y * np.log(a+1e-3)  # log(a + 1e-3)
    p2 = (1-y) * np.log(1-a+1e-3)
    loss = -p1 - p2
    return loss

def my_loss(a, y):
    return y * (np.exp(1-a)-1) + (1-y) * (np.exp(a)-1)


def de_my_loss(a, y):
    return -y * np.exp(1-a) + (1 - y) * np.exp(a)

def draw_loss():
    x = np.linspace(0, 1, 20)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    y = 1
    loss = my_loss(x, y)
    ax.plot(x, loss, label="自定义损失函数")
    loss = bce_loss_function(x, y)
    ax.plot(x, loss, linestyle="dashed", label="交叉熵损失函数")
    ax.grid()
    ax.set_xlabel("a")
    ax.set_ylabel("loss")
    ax.set_title("正类的损失函数")
    ax.legend()

    ax = fig.add_subplot(1,2,2)
    y = 0
    loss = my_loss(x, y)
    ax.plot(x, loss, label="自定义损失函数")
    loss = bce_loss_function(x, y)
    ax.plot(x, loss, linestyle="dashed", label="交叉熵损失函数")
    ax.grid()
    ax.set_xlabel("a")
    ax.set_ylabel("loss")
    ax.set_title("负类的损失函数")
    ax.legend()

    plt.show()

def my_activator(x):
    return x / (1 + np.abs(x))

def de_my_activator(x):
    return 1 / np.square(1 + np.abs(x))


def tanh(x):
    return 2 / (1 + np.exp(-2*x)) - 1

def de_tanh(x):
    return 1 - np.square(tanh(x))

def draw_activator():
    x = np.linspace(-5, 5, 51)
    plt.plot(x, tanh(x), linestyle="dashed", label="Tanh激活函数")
    plt.plot(x, de_tanh(x), linestyle="dashed", marker='.', label="Tanh导数")

    plt.plot(x, my_activator(x), label="自定义激活函数")
    plt.plot(x, de_my_activator(x), marker='.', label="自定义激活函数导数")
    plt.grid()
    plt.legend()
    plt.show()


def logistic(x):
    return 1 / (1 + np.exp(-x))

def my_classifier(x):
    return 0.5 + np.arctan(x) / np.pi

def de_my_classifier(x):
    return 1 / (np.pi * (1 + np.square(x)))



def draw_classifier():
    x = np.linspace(-5, 5, 100)
    y1 = logistic(x)
    y2 = my_classifier(x)
    plt.plot(x, y1, linestyle="dashed", label="Logit")
    plt.plot(x, y2, label="自定义分类函数")
    plt.legend()
    plt.grid()
    plt.show()

if __name__=="__main__":
    draw_loss()
    draw_activator()
    draw_classifier()

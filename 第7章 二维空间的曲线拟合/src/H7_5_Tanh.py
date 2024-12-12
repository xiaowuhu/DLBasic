
import numpy as np
from common.Activators import Tanh
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def f1(x):
    return Tanh().forward(x + 1.5)

def f2(x):
    return Tanh().forward(-2 * x + 2)

def f3(x):
    return Tanh().forward(3 * x - 3)

def f4(x):
    return Tanh().forward(-4 * x -4)

def set_ax(ax, title):
    ax.set_xlim((-3,3))
    ax.set_ylim((-3,3))
    ax.plot([-3,3],[3,-3], c='gray')
    ax.set_title(title)
    ax.grid()


if __name__=="__main__":
    x = np.linspace(-3, 3, 100)[:, np.newaxis]
    y1 = f1(x)
    y2 = f2(x)
    y3 = f3(x)
    y4 = f4(x)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, y1)
    set_ax(ax, "$y_1$=tanh($x+1.5$)")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, y2)
    set_ax(ax, "$y_2$=tanh($-2x+2$)")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, y3)
    set_ax(ax, "$y_3$=tanh($3x-3$)")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, y4)
    set_ax(ax,"$y_4$=tanh($-4x-4$)")

    plt.show()


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(x, y1 + y2)
    set_ax(ax, "$y_1+y_2$")
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(x, y1 + y3)
    set_ax(ax, "$y_1+y_3$")
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(x, y3 + y4)
    set_ax(ax, "$y_3 + y_4$")
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(x, y2 + y4)
    set_ax(ax, "$y_2+y_4$")
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(x, y3*(y1 + y2))
    set_ax(ax, "$y_3(y_1+y_2)$")
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(x, y4*(y1 + y2))
    set_ax(ax, "$y_4(y_1+y_2)$")
    plt.show()    

import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

np.set_printoptions(suppress=True, threshold=sys.maxsize)

# Logistic 函数
def logistic(z):
    a = 1.0 / (1.0 + np.exp(-z))
    return a    

# Logistic 函数的导数
def logistic_derivative(z):
    # dz = np.multiply(a, 1-a)
    tmp = np.exp(-z)
    dz = tmp / (1 + tmp)**2
    return dz

def show_logistic():
    x = np.linspace(-10, 10, 100)
    y = logistic(x)
    plt.plot(x, y, label="函数")
    y = logistic_derivative(x)
    plt.plot(x, y, linestyle=":", label="导数")
    plt.grid()
    plt.legend()
    plt.show()

# 对数几率函数
def show_odds():
    p = np.linspace(0.1, 0.9, 9)
    q = 1 - p
    print(p)
    print(q)
    print(p/q)
    print(np.log(p/q))
    plt.plot(p, p, label="p", marker='o')
    plt.plot(p, q, label="q", marker='*')
    plt.plot(p, p/q, label="odds", marker='v')
    plt.plot(p, np.log(p/q), label="ln(odds)", marker='s')
    plt.grid()
    plt.legend()
    plt.xlabel("p")
    plt.show()

if __name__=="__main__":
    show_odds()
    show_logistic()

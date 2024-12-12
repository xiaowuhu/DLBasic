import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data = np.loadtxt(filename)
    return data

def show_data(n1:int=None, n2:int=None, W=None, B=None):
    train_data = load_data("train5.txt")
    x1 = train_data[train_data[:, 3]==1]
    x2 = train_data[train_data[:, 3]==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[0:n1, 0], x1[0:n1, 1], c='r', marker='+', label='学区房')
    plt.scatter(x2[0:n2, 0], x2[0:n2, 1], c='b', marker='.', label='普通房')
    plt.grid()
    plt.legend(loc='upper right')
    if W is not None:
        assert(B is not None)
        w = - W[0, 0] / W[1, 0]
        b = - B[0, 0] / W[1, 0]
        x = np.linspace(0, 0.5, 2)
        y = w * x + b
        plt.plot(x, y)
        plt.axis([-0.1, 1.1, -0.1, 1.1])

    plt.show()
    return train_data

if __name__=="__main__":
    train_data = show_data(50, 200)
    index = np.random.choice(np.arange(100), 10)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    print(train_data[index])


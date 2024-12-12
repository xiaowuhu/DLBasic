import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from common.DataLoader_8 import DataLoader_8
from matplotlib.colors import LogNorm

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data_loader = DataLoader_8(filename)
    data_loader.load_data()
    return data_loader

def show_data(data_loader: DataLoader_8):
    X, Y = data_loader.get_train()
    fig = plt.figure()
    x = X[:, 0]
    y = X[:, 1]
    z = Y
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.grid()
    plt.show()

def get_min_max(x):
    return np.min(x), np.max(x), np.mean(x), np.var(x)

if __name__=="__main__":
    dl = load_data("train8.txt")
    show_data(dl)

    X, Y = dl.get_train()
    print("--- X", X.shape, "---")
    print("最大值:", np.max(X, axis=0))
    print("最小值:", np.min(X, axis=0))
    print("均值:", np.mean(X, axis=0))
    print("标准差:", np.std(X, axis=0))
    print("--- Y", Y.shape, "---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("均值:", np.mean(Y))
    print("标准差:", np.std(Y))

    dl.StandardScaler_X()
    dl.StandardScaler_Y()

    print("====== 标准化后的统计信息 ======")
    X, Y = dl.get_train()
    print("--- X", X.shape, "---")
    print("最大值:", np.max(X, axis=0))
    print("最小值:", np.min(X, axis=0))
    print("均值:", np.mean(X, axis=0))
    print("标准差:", np.std(X, axis=0))
    print("--- Y", Y.shape, "---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("均值:", np.mean(Y))
    print("标准差:", np.std(Y))


import numpy as np
import os
import matplotlib.pyplot as plt
from common.DataLoader_11 import DataLoader_11

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir,  "data", name)
    data_loader = DataLoader_11(filename)
    data_loader.load_data()
    return data_loader

def show_data(X, Y):
    X0 = X[Y[:,0]==0]
    X1 = X[Y[:,0]==1]
    plt.figure(figsize=(5, 5))
    plt.scatter(X0[:,0], X0[:,1], marker='o', label="0", s=20)
    plt.scatter(X1[:,0], X1[:,1], marker='x', label="1", s=20)
    plt.grid()
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__=="__main__":
    data_loader = load_data("train11-moon.txt")
    X, Y = data_loader.get_train()
    show_data(X, Y)
    np.set_printoptions(precision=3)
    print("--- X", X.shape, "---")
    print("最大值:", np.max(X, axis=0))
    print("最小值:", np.min(X, axis=0))
    print("均值:", np.mean(X, axis=0))
    print("标准差:", np.std(X, axis=0))
    print("--- Y", Y.shape, "---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("分类值:", np.unique(Y))

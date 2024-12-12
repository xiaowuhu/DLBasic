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
    filename = os.path.join(current_dir, "data",  name)
    data_loader = DataLoader_11(filename)
    data_loader.load_data()
    return data_loader

def show_data(X, Y):
    X0 = X[Y[:,0]==0]
    X1 = X[Y[:,0]==1]
    X2 = X[Y[:,0]==2]
    X3 = X[Y[:,0]==3]
    plt.figure(figsize=(5, 5))
    plt.scatter(X0[:,0], X0[:,1], marker='o', label="0", s=20)
    plt.scatter(X1[:,0], X1[:,1], marker='x', label="1", s=20)
    plt.scatter(X2[:,0], X2[:,1], marker='.', label="2", s=20)
    plt.scatter(X3[:,0], X3[:,1], marker='^', label="3", s=20)
    plt.grid()
    plt.legend(loc="right")
    plt.axis("equal")
    plt.show()


if __name__=="__main__":
    data_loader = load_data("train11-taiji.txt")
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

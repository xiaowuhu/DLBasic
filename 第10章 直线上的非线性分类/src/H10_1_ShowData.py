import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from common.DataLoader_10 import DataLoader_10

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data_loader = DataLoader_10(filename)
    data_loader.load_data()
    data_loader.shuffle_data()
    return data_loader

def show_data(X, Y):
    X0 = X[Y==0][:20]
    X1 = X[Y==1][:20]
    plt.scatter(X0, [0]*len(X0), marker='.', label="0-噪音污染较大")
    plt.scatter(X1, [0]*len(X1), marker='x', label="1-噪音污染较小")
    plt.xlabel("高度（米）")
    plt.grid()
    plt.legend()
    plt.show()

if __name__=="__main__":
    data_loader = load_data("train10.txt")
    X, Y = data_loader.get_train()
    show_data(X, Y)
    np.set_printoptions(precision=3)
    
    print("--- X (%d) ---"%X.shape[0])
    print("最大值:", np.max(X))
    print("最小值:", np.min(X))
    print("均值:", np.mean(X))
    print("标准差:", np.std(X))
    print("--- Y (%d) ---"%Y.shape[0])
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("均值:", np.mean(Y))
    print("标准差:", np.std(Y))

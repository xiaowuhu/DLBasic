import os
import numpy as np
import matplotlib.pyplot as plt
from common.DataLoader_7 import DataLoader_7

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data_loader = DataLoader_7(filename)
    data_loader.load_data()
    return data_loader

def show_data(data_loader: DataLoader_7):
    X, Y = data_loader.get_train()
    plt.scatter(X, Y, s=1)
    plt.xlabel("时间（年）")
    plt.ylabel("单价（万元/平米）")
    plt.grid()
    plt.show()

if __name__=="__main__":
    dl = load_data("train7.txt")
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

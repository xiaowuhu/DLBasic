import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from common.DataLoader_9 import DataLoader_9

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_name, val_name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename1 = os.path.join(current_dir, "data", train_name)
    filename2 = os.path.join(current_dir, "data", val_name)
    data_loader = DataLoader_9(filename1, filename2)
    data_loader.load_data()
    return data_loader

def show_data(X, Y):
    plt.scatter(X, Y)
    plt.xlabel("区域长度(百米)")
    plt.ylabel("区域宽度(百米)")
    plt.grid()
    X, Y = data_loader.get_val()
    plt.plot(X, Y)
    plt.show()

if __name__=="__main__":
    data_loader = load_data("train9.txt", "val9.txt")
    X, Y = data_loader.get_train()
    show_data(X, Y)
    np.set_printoptions(precision=3)
    print("--- X ---")
    print("最大值:", np.max(X))
    print("最小值:", np.min(X))
    print("均值:", np.mean(X))
    print("标准差:", np.std(X))
    print("--- Y ---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("均值:", np.mean(Y))
    print("标准差:", np.std(Y))

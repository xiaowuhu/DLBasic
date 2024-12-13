import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from common.DataLoader_13 import DataLoader_13

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_13(train_file_path, test_file_path)
    data_loader.load_data()
    data_loader.shuffle_data()
    return data_loader

def show_class_data(class_id, ax, label):
    for i in range(X.shape[0]):
        if Y[i] == class_id:
            x = np.linspace(0,1,X[i].shape[0])
            ax.plot(x, X[i], marker='.', label=label)
            ax.legend()
            break

def show_data(X, Y):
    fig = plt.figure(figsize=(12,5))
    labels = ["sin","cos","sawtooth","flat","-sin","-cos","-sawtooth","-flat"]
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1)
        show_class_data(i, ax, labels[i])
        ax.grid()
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.4, 1.4)
        
    plt.show()

if __name__=="__main__":
    data_loader = load_data("train13.txt", "test13.txt")
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
    print("分类标签:", np.unique(Y).astype(int))

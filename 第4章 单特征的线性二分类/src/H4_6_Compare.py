import os
from common.DataLoader_4 import DataLoader_4
from H4_1_ShowData import load_data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train4.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_4(file_path)
    data_loader.load_data([0, 1])
    data_loader.split_data(0.8)
    return data_loader

# n1 - 正类样本数量，n2 - 负类样本数量
def show_result(X, Y, n1, n2, w = None, b = None, x = None):
    x1 = X[Y==1] 
    x2 = X[Y==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[0:n1], [0]*n1, c='r', marker='x')
    plt.scatter(x2[0:n2], [0]*n2, c='b', marker='o')
    plt.xlabel("w")
    plt.ylabel("b")
    plt.grid()

if __name__ == '__main__':
    weight = [-13.56, -22.93, -28.26]
    bias = [21.29, 34.62, 42.16]
    data_loader = load_data()
    X, Y = data_loader.get_train()
    show_result(X, Y, 20, 30)
    linestyles = ["-", "-.", ":"]
    for i in range(3):
        x = - bias[i] / weight[i]
        # 画出分界线
        minv = np.min(X)
        maxv = np.max(X)
        plt.plot(
            [minv, maxv], 
            [minv * weight[i] + bias[i], maxv * weight[i] + bias[i]], 
            label="x=%.2f"%x, 
            linestyle=linestyles[i])
    plt.legend()
    plt.show()
        

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
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

# n1 - 正类样本数量，n2 - 负类样本数量
def show_result(X, Y, n1, n2):
    x1 = X[Y==1] 
    x2 = X[Y==0]
    plt.scatter(x1[0:n1], [0]*n1, c='r', marker='x')
    plt.scatter(x2[0:n2], [0]*n2, c='b', marker='o')

if __name__ == '__main__':
    data_loader = load_data()
    X, Y = data_loader.get_train()
    show_result(X, Y, 20, 30)
    W = [-13.56, -22.93, -28.26]
    B = [21.29, 34.62, 42.16]
    LineStyles = ['-', '--', '-.']
    for i in range(3):
        x_0 = -B[i] / W[i]
        plt.scatter(x_0, 0, marker="*", s=50)
        # 画出分界线
        minv = np.min(X)
        maxv = np.max(X)
        plt.plot([minv,maxv], [minv*W[i]+B[i],maxv*W[i]+B[i]], linestyle=LineStyles[i], label="$x_0=%.2f$"%x_0)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

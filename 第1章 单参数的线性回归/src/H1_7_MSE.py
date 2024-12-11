import os
import numpy as np
from common.NeuralNet_1_7 import NeuralNet_1_7
from common.DataLoader_1 import DataLoader_1
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 计算均方误差
def MSE(X, Y, w):
    z = X * w
    mse = np.mean((z - Y) ** 2) / 2
    return mse

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train1.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_1(file_path)
    data_loader.load_data()

    X = data_loader.data[0]
    Y = data_loader.data[1]
    mse = MSE(X, Y, 2.0158)
    print(mse)

    W = np.linspace(2.013, 2.27, 100)  # 100 等分
    MSEs = []  # 存储 100 个不同的 mse 值
    for w in W:
        mse = MSE(X, Y, w)
        MSEs.append(mse)
    plt.plot(W, MSEs, linestyle="solid", label="全体样本的 MSE")
    plt.scatter(2.0158, 236.626)
    plt.xlabel("w")
    plt.ylabel("mse")
    plt.grid()

    W = np.linspace(2.013, 2.27, 100)  # 100 等分
    MSEs = []  # 存储 100 个不同的 mse 值
    for w in W:
        mse = MSE(121, 237, w)
        MSEs.append(mse)
    plt.plot(W, MSEs, linestyle="dotted", label="样本 1 的 loss")
    plt.scatter(2.0158, 25)
    plt.legend()
    plt.show()


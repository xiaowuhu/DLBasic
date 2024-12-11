import os
import numpy as np
from common.NeuralNet_1_7 import NeuralNet_1_7
from common.DataLoader_1 import DataLoader_1
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

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

    X, Y = data_loader.get_data()
    mse = MSE(X, Y, 2.0158)
    print(mse)

    W = np.linspace(1.35, 3, 100)  # 100 等分
    MSEs = []  # 存储 100 个不同的 mse 值
    for w in W:
        mse = MSE(X, Y, w) # 计算 100 个不同的 w 对应的均方误差MSE
        MSEs.append(mse)

    W = np.linspace(1.35, 3, 100)  # 100 等分
    X, Y = data_loader.get_data()
    Error_array = np.zeros((10, 100)) # 10个样本对应100个w的平方误差
    for i in range(X.shape[0]): # 计算10个样本的平方误差
        x = X[i]
        y = Y[i]
        for j, w in enumerate(W):
            error = (x * w - y)**2 / 2
            Error_array[i, j] = error
        if i == 0:
            plt.plot(W, Error_array[i], linestyle="dotted", label="十个独立样本的平方误差")
        else:
            plt.plot(W, Error_array[i], linestyle="dotted")

    mse = np.mean(Error_array, axis=0)
    plt.plot(W, mse, label="全体样本的均方误差 MSE")
    plt.legend()
    plt.xlabel("$w$")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

    MSEs = np.array(MSEs)
    print(np.allclose(MSEs, mse))

    

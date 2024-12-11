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
    check_data = np.linspace(2.013, 2.03, 100)  # 100 等分
    MSEs = []  # 存储 100 个不同的 mse 值
    for w in check_data:
        mse = MSE(X, Y, w)
        MSEs.append(mse)
    plt.plot(check_data, MSEs)
    plt.grid()
    #plt.show()
    
    print("训练神经网络...")
    lr = 1e-6 # 0.000001
    w = 0
    epoch = 1000
    nn = NeuralNet_1_7(data_loader, w, lr)
    check_data = nn.train(epoch, checkpoint=30)
    print(check_data)
    check_data = np.array(check_data)
    for i in range(check_data.shape[0]):
        w = check_data[i, 0]
        dw = check_data[i, 1]
        x = check_data[i, 2]
        y = check_data[i, 3]
        mse = MSE(X, Y, w)
        text = str.format("{0}:w={1:.4f},dw={2:.1f},x={3},y={4}", i+1, w, dw, x, y)
        plt.scatter(w, mse)
        plt.text(w+0.001, mse, text, fontsize=10)
    plt.xlabel("w")
    plt.ylabel("mse")
    plt.legend()
    plt.show()


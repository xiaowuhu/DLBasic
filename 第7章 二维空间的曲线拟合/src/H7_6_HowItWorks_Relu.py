import os
from common.NeuralNet_7 import NeuralNet_7
from common.DataLoader_7 import DataLoader_7
import matplotlib.pyplot as plt
import numpy as np
from common.Activators import Relu

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_7(file_path)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    data_loader.shuffle_data()
    return data_loader

def load_result():
    file_name = "weight-bias-1-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P1 = np.loadtxt(file_path)
    W1 = P1[0:-1]
    B1 = P1[-1:]
    file_name = "weight-bias-2-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P2 = np.loadtxt(file_path)[:, np.newaxis]
    W2 = P2[0:-1]
    B2 = P2[-1:]
    return W1, B1, W2, B2

def show_result(nn):
    X, Y = data_loader.get_train()
    X = data_loader.de_MinMaxScaler_X(X)
    Y = data_loader.de_MinMaxScaler_Y(Y)
    plt.scatter(X, Y, s=1)
    plt.xlabel("时间（年）")
    plt.ylabel("房屋均价（万元/平米）")
    X = np.linspace(0, 7, 100)[:, np.newaxis]
    normalized_X = data_loader.MinMaxScaler_pred_X(X)
    normalized_Y = nn.predict(normalized_X)
    Y = data_loader.de_MinMaxScaler_Y(normalized_Y)
    plt.plot(X, Y)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data("train7.txt")
    params = load_result()
    nn = NeuralNet_7(data_loader, *params)

    grid = plt.GridSpec(nrows=2, ncols=4)
    plt.figure(figsize=(15, 6))
    # 原始直线 20 个点
    x = np.linspace(0, 1, 20)[:, np.newaxis]
    plt.subplot(grid[0:2, 0])
    plt.plot(x, [0]*20, marker='.')
    plt.title("原始样本数据")
    plt.grid()
    # 隐层
    z1 = np.dot(x, nn.W1) + nn.B1
    # 神经元1
    plt.subplot(grid[0:1, 1])
    plt.plot(x, z1[:, 0], marker='x')
    title = str.format("z_1={0:.2f}x+{1:.2f}", nn.W1[0,0], nn.B1[0,0])
    plt.title(r"神经元1 : $" + title + "$")
    plt.grid()
    # 神经元2
    plt.subplot(grid[1:2, 1])
    plt.plot(x, z1[:, 1], marker='o')
    title = str.format("z_2={0:.2f}x+{1:.2f}", nn.W1[0,1], nn.B1[0,1])
    plt.title(r"神经元2 : $" + title + "$")
    plt.grid()
    # 激活输出
    tanh = Relu()
    a1 = tanh.forward(z1)
    # 神经元1
    plt.subplot(grid[0:1, 2])
    plt.title("神经元1 : $a_1=$relu$(z_1)$")
    plt.plot(x, a1[:, 0], marker='x')
    plt.grid()
    # 神经元2
    plt.subplot(grid[1:2, 2])
    plt.title("神经元2 : $a_2=$relu$(z_2)$")
    plt.plot(x, a1[:, 1], marker='o')
    plt.grid()
    z2 = np.dot(a1, nn.W2) + nn.B2
    plt.subplot(grid[0:2, 3])
    plt.plot(x, z2, marker='.') 
    plt.grid() 
    title = str.format("z={0:.2f} a_1+{1:.2f} a_2 + {2:.2f}", nn.W2[0,0],nn.W2[1,0], nn.B2[0,0])
    plt.title(r"$" + title + "$")
    plt.show()

    np.set_printoptions(precision=4)
    print("原始样本点(20个):")
    print(x.transpose())
    print("隐层线性变换输出 z1,z2:")
    print(z1.transpose())
    print("隐层激活输出 a1,a2:")
    print(a1.transpose())
    print("最终回归输出 z:")
    print(z2.transpose())
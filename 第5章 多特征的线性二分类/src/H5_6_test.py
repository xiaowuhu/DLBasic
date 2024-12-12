import os
import sys
from common.NeuralNet_5_6 import NeuralNet_5
from common.DataLoader_5 import DataLoader_5
from H5_1_ShowData import show_data, load_data
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_5 import TrainingHistory_5

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

W1_POS = 0
W2_POS = 1
W3_POS = 2
B_POS = 3
DW1_POS = 4
DW2_POS = 5
DW3_POS = 6
DB_POS = 7

def load_data():
    file_name = "train5.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_5(file_path)
    data_loader.load_data([0, 1, 5, 3]) # x, y, 价格, 学区房标签
    data_loader.normalize_train_data()  # 需要归一化价格数据
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_5):
    batch_size = 10
    epoch = 100
    lr = 0.5
    W = np.zeros((3,1))
    B = np.zeros((1,1))
    nn = NeuralNet_5(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history, history = nn.train(epoch, checkpoint=10)
    # np.savetxt("history_5_6.txt", history, fmt="%.6f")
    return nn, training_history

def show_result(data_loader, W, B):
    X,Y  = data_loader.get_train()
    x1 = X[Y[:,0]==1]
    x2 = X[Y[:,0]==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='+', label='学区房')
    plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='.', label='普通房')
    plt.grid()
    plt.legend(loc='upper right')
    if W is not None:
        assert(B is not None)
        w = - W[0, 0] / W[1, 0]
        b = - B[0, 0] / W[1, 0]
        x = np.linspace(0, 1, 2)
        y = w * x + b
        plt.plot(x, y)
        plt.axis([-0.1, 1.1, -0.1, 1.1])

    plt.show()

# 平滑处理
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    re = np.convolve(interval, window, 'same')
    return re

if __name__ == '__main__':
    # 由以下代码训练网络得到真实数据并保存
    # print("加载数据...")
    #data_loader = load_data()
    # print("训练神经网络...")
    #nn, training_history = train(data_loader)
    # iteration, val_loss, W, B = training_history.get_best()
    # print("权重值 w =\n", W)
    # print("偏置值 b =\n", B)
    # training_history.show_loss()
    # show_result(data_loader, W, B)

    # 画图
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_5_6.txt")
    history = np.loadtxt(file_path)
    mean = np.mean(history, axis=1)
    print("均值 dW1:", mean[4], "dW2:", mean[5], "dW3:", mean[6])
    # print("W1:", history[W1_POS,0:10])
    # print("W2:", history[W2_POS,0:10])
    # print("W3:", history[W3_POS,0:10])
    # print("dW1:", history[DW1_POS,0:10])
    # print("dW2:", history[DW2_POS,0:10])
    # print("dW3:", history[DW3_POS,0:10])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2,1,1)
    ax.plot(history[0], label="$w_1$")
    ax.plot(history[1], label="$w_2$", marker='x', markevery=0.1)
    ax.plot(history[2], label="$w_3$", marker='o', markevery=0.1)
    ax.legend()
    ax.set_title("权重值")
    ax.grid()
    ax = fig.add_subplot(2,1,2)
    av4 = moving_average(history[4], 2000)
    ax.plot(av4, label="$dw_1$:%f"%mean[4])
    av5 = moving_average(history[5], 2000)
    ax.plot(av5, label="$dw_2$:%f"%mean[5], marker='x', markevery=0.1)
    av6 = moving_average(history[6], 2000)
    ax.plot(av6, label="$dw_3$:%f"%mean[6], marker='o', markevery=0.1)
    ax.legend()
    ax.set_title("权重梯度")
    ax.grid()


    plt.show()


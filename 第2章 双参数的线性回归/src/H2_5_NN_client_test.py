import os
from common.NeuralNet_2_test import NeuralNet_2
from common.DataLoader_2 import DataLoader_2
import matplotlib.pyplot as plt
from H2_2_MSE import prepare_data
from H2_3_LeastSquare import method1, calculate_b_2
from matplotlib.colors import LogNorm
import numpy as np

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()
    print("训练神经网络...")
    lr = 0.00001
    w = 0
    b = 0
    epoch = 100000
    nn = NeuralNet_2(data_loader, w, b, lr)
    W, B = nn.train_test(epoch)
    print(W)
    print(B)
    # 绘制 w,b 轨迹
    plt.plot(W, B)

    X = data_loader.data[0]
    Y = data_loader.data[1]
    # 得到准确值作为绘图参考基准
    w_truth = method1(X, Y, X.shape[0])
    b_truth = calculate_b_2(X, Y, w_truth)
    W, B, Loss = prepare_data(X, Y, w_truth, b_truth, 2, 50, 1000, 1000)
    plt.contour(W, B, Loss, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)
    plt.xlabel("w")
    plt.ylabel("b")
    plt.grid()
    plt.show()


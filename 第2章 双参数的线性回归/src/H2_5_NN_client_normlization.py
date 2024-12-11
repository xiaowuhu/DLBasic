import os
from common.NeuralNet_2_test import NeuralNet_2
from common.DataLoader_2 import DataLoader_2
import matplotlib.pyplot as plt
from H2_2_MSE import prepare_data
from H2_3_LeastSquare import calculate_w_b
from matplotlib.colors import LogNorm
import numpy as np


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()
    data_loader.normalize_train_data()
    print("训练神经网络...")
    lr = 0.01
    w = 0
    b = 0
    epoch = 200
    nn = NeuralNet_2(data_loader, w, b, lr)
    W, B = nn.train_test(epoch, checkpoint=2, add_start=True)
    print("权重值 w = %f, 偏置值 b = %f" % (nn.w, nn.b))
    # 绘制 w,b 轨迹
    plt.plot(W, B, "ob:")
    # 预测
    pred_x = 120
    normlized_x = data_loader.normalize_pred_data(pred_x) # 归一化
    pred_y = nn.predict(normlized_x) # 预测
    y = data_loader.de_normalize_y_data(pred_y) # 反归一化
    print("预测：面积为 %f 的房屋价格约为 %f" % (pred_x, y))
    # 绘制损失函数等高线
    X = data_loader.train_x
    Y = data_loader.train_y
    # 得到准确值作为绘图参考基准
    w_truth, b_truth = calculate_w_b(X, Y)
    W, B, Loss = prepare_data(X, Y, w_truth, b_truth, 1, 0.5, 100, 100)
    plt.contour(W, B, Loss, levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap="Greys")
    plt.xlabel("w")
    plt.ylabel("b")
    plt.grid()
    plt.show()


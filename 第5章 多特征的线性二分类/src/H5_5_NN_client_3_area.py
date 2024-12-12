import os
import sys
from common.NeuralNet_5 import NeuralNet_5
from common.DataLoader_5 import DataLoader_5
from H5_1_ShowData import show_data, load_data
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_5 import TrainingHistory_5
from H5_5_NN_client_1_sun import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train5.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_5(file_path)
    data_loader.load_data([0, 1, 2, 3]) # x, y, 面积, 学区房标签
    data_loader.normalize_train_data()  # 归一化面积数据
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

if __name__ == '__main__':
    np.set_printoptions(suppress=True, threshold=sys.maxsize)

    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history = train(data_loader)
    iteration, val_loss, W, B = training_history.get_best()
    print("权重值 w =\n", W)
    print("偏置值 b =\n", B)
    training_history.show_loss()
    show_result(W, B)

    print("---- 测试 ----")
    x = np.array([[0.2, 0.8, 50], [0.2, 0.8, 120], [0.5, 0.6, 50], [0.5, 0.6, 120]])
    pred_x = data_loader.normalize_pred_data(x)
    print("---- 归一化结果 ----")
    print(pred_x)
    pred_y = nn.forward(pred_x)
    print("---- 分类值 ----")
    print(pred_y)
    print("---- 结论 ----")
    for i in range(pred_x.shape[0]):
        if pred_y[i] > 0.5:
            print("样本", x[i], "的房子预测值", pred_y[i], "是学区房。")
        else:
            print("样本", x[i], "的房子预测值", pred_y[i], "不是学区房。")
    
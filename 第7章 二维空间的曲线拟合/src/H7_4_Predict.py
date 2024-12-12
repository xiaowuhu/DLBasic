import os
from common.NeuralNet_7 import NeuralNet_7
from common.DataLoader_7 import DataLoader_7
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_7 import TrainingHistory_7

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_7(file_path)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    return data_loader

def load_result():
    file_name = "weight-bias-1-tanh.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P1 = np.loadtxt(file_path)
    W1 = P1[0:-1]
    B1 = P1[-1:]
    file_name = "weight-bias-2-tanh.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P2 = np.loadtxt(file_path)
    W2 = P2[0:-1]
    B2 = P2[-1:]
    return W1, B1, W2, B2

def show_result(nn):
    # 绘制样本数据
    X, Y = data_loader.get_train()
    X = data_loader.de_MinMaxScaler_X(X)
    Y = data_loader.de_MinMaxScaler_Y(Y)
    plt.scatter(X, Y, s=1)
    plt.xlabel("时间（年）")
    plt.ylabel("房屋均价（万元/平米）")
    # 绘制拟合曲线
    X = np.linspace(0, 7, 100)[:, np.newaxis]
    normalized_X = data_loader.MinMaxScaler_pred_X(X)
    normalized_y = nn.predict(normalized_X)
    Y = data_loader.de_MinMaxScaler_Y(normalized_y)
    plt.plot(X, Y)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data("train7.txt")
    params = load_result()
    nn = NeuralNet_7(data_loader, *params)
    print("权重值 w1 =", nn.W1)
    print("偏置值 b1 =", nn.B1)
    print("权重值 w2 =", nn.W2)
    print("偏置值 b2 =", nn.B2)
    print("预测...")
    #              2年2个月     5年7个月
    X = np.array([[2 + 2 / 12],[5 + 7 / 12]])
    normalized_X = data_loader.MinMaxScaler_pred_X(X)
    normalized_y = nn.predict(normalized_X)
    Y = data_loader.de_MinMaxScaler_Y(normalized_y)
    print("第二年零两个月商品房的平均单价:%.2f 万元/平米" %(Y[0]))
    print("第五年零七个月商品房的平均单价:%.2f 万元/平米" %(Y[1]))

    show_result(nn)

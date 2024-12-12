import os
from common.NeuralNet_7_6 import NeuralNet_7_6
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
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_7):
    batch_size = 8
    epoch = 2000
    lr = 0.1
    num_hidden = 2
    W1 = np.random.normal(size=(1, num_hidden))
    B1 = np.random.normal((1, num_hidden))
    W2 = np.random.normal(size=(num_hidden, 1))
    B2 = np.zeros((1, 1)) + 0.1
    nn = NeuralNet_7_6(data_loader, W1, B1, W2, B2, lr=lr, batch_size=batch_size)
    history = nn.train(epoch, checkpoint=10)
    print(W1)
    print(B1)
    print(W2)
    print(B2)

    return nn, history

def save_result(nn):
    P1 = np.concatenate((nn.W1, nn.B1))
    file_name = "weight-bias-1-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, P1, fmt="%f")
    P2 = np.concatenate((nn.W2, nn.B2))
    file_name = "weight-bias-2-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, P2, fmt="%f")

def show_result(nn, data_loader: DataLoader_7):
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
    normalized_Y = nn.predict(normalized_X)
    Y = data_loader.de_MinMaxScaler_Y(normalized_Y)
    plt.plot(X, Y)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data("train7.txt")
    print("训练神经网络...")
    nn, history = train(data_loader)
    print("训练完成")
    print("权重值 w1 =", nn.W1)
    print("偏置值 b1 =", nn.B1)
    print("权重值 w2 =", nn.W2)
    print("偏置值 b2 =", nn.B2)
    # save_result(nn)
    show_result(nn, data_loader)
    history.show_loss()

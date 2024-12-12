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
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def init_weights(w, b):
    num_hidden = 2
    if w == "00":
        W1 = np.zeros((1, num_hidden))
        W2 = np.zeros((num_hidden, 1))
    elif w == "01":
        W1 = np.zeros((1, num_hidden))
        W2 = np.ones((num_hidden, 1))
    elif w == "10":
        W1 = np.ones((1, num_hidden))
        W2 = np.zeros((num_hidden, 1))
    elif w == "11":
        W1 = np.ones((1, num_hidden))
        W2 = np.ones((num_hidden, 1))
    elif w == "0n":
        W1 = np.zeros((1, num_hidden))
        W2 = np.random.normal(size=(num_hidden, 1))
    elif w == "n0":
        W1 = np.random.normal(size=(num_hidden, 1))
        W2 = np.zeros((num_hidden, 1))
    else:  # nn
        W1 = np.random.normal(size=(1, num_hidden))
        W2 = np.random.normal(size=(num_hidden, 1))

    if b == "00":
        B1 = np.zeros((1, num_hidden))
        B2 = np.zeros((1, 1))
    elif b == "01":
        B1 = np.zeros((1, num_hidden))
        B2 = np.ones((1, 1))
    elif b == "10":
        B1 = np.ones((1, num_hidden))
        B2 = np.zeros((1, 1))
    elif b == "11":
        B1 = np.ones((1, num_hidden))
        B2 = np.ones((1, 1))
    elif b == "0n":
        B1 = np.zeros((1, num_hidden))
        B2 = np.random.normal(size=(1, 1))
    elif b == "n0":
        B1 = np.random.normal(size=(1, num_hidden))
        B2 = np.zeros((1, 1))
    else: # nn, 0n, n0, 1n, n1
        B1 = np.random.normal(size=(1, num_hidden))
        B2 = np.random.normal(size=(1, 1))

    return W1, B1, W2, B2

def train(data_loader: DataLoader_7):
    batch_size = 32
    epoch = 1000
    lr = 0.5
    W1, B1, W2, B2 = init_weights("00", "0n")
    nn = NeuralNet_7(data_loader, W1, B1, W2, B2, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=10)
    return nn, training_history


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
    print("训练神经网络...")
    nn, training_history = train(data_loader)
    print("权重值 w1 =", nn.W1)
    print("偏置值 b1 =", nn.B1)
    print("权重值 w2 =", nn.W2)
    print("偏置值 b2 =", nn.B2)
    training_history.show_loss()
    show_result(nn)

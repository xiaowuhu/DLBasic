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

def train(data_loader: DataLoader_7):
    batch_size = 32
    epoch = 2000
    lr = 0.5
    num_hidden = 2
    W1 = np.random.normal(size=(1, num_hidden))
    B1 = np.random.normal(size=(1, num_hidden))
    W2 = np.random.normal(size=(num_hidden, 1))
    B2 = np.random.normal(size=(1, 1))
    nn = NeuralNet_7(data_loader, W1, B1, W2, B2, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=10)
    return nn, training_history

def save_result(nn):
    P1 = np.concatenate((nn.W1, nn.B1))
    file_name = "weight-bias-1-tanh.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, P1, fmt="%f")
    P2 = np.concatenate((nn.W2, nn.B2))
    file_name = "weight-bias-2-tanh.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, P2, fmt="%f")


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
    # save_result(nn)
    training_history.show_loss()

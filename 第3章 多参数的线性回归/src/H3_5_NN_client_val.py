import os
from common.NeuralNet_3_5 import NeuralNet_3_5
from common.DataLoader_3_5 import DataLoader_3_5
from H3_1_ShowData import load_data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def load_data():
    file_name = "train3.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_3_5(file_path)
    data_loader.load_data()
    data_loader.normalize_train_data()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_3_5):
    batch_size = 10
    epoch = 300
    lr = 0.01
    W = np.zeros((2,1))
    B = np.zeros((1,1))
    nn = NeuralNet_3_5(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=80)
    return nn, training_history

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history = train(data_loader)
    iteration, val_loss, W, B = training_history.get_best()
    print("权重值 w =", W)
    print("偏置值 b =", B)
    training_history.show_loss(start=0)

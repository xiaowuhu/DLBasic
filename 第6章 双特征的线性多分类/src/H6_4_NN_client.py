import os
from common.NeuralNet_6 import NeuralNet_6
from common.DataLoader_6 import DataLoader_6
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2]) # 加载 横坐标，纵坐标，分类标签
    data_loader.to_onehot(3) # 标签变为 onehot
    # data_loader.normalize_train_data()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_6):
    batch_size = 32
    epoch = 1000
    lr = 0.5
    W = np.zeros((2,3))
    B = np.zeros((1,3))
    nn = NeuralNet_6(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=80)
    return nn, training_history

def save_result(nn):
    file_name = "weights6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, nn.W)
    file_name = "bias6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, nn.B)


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history = train(data_loader)
    iteration, val_loss, W, B = training_history.get_best()
    print("权重值 w =", W)
    print("偏置值 b =", B)
    training_history.show_loss()
    # save_result(nn)

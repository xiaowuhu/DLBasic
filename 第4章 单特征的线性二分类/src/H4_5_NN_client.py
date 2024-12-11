import os
from common.NeuralNet_4 import NeuralNet_4
from common.DataLoader_4 import DataLoader_4
from H4_1_ShowData import load_data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train4.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_4(file_path)
    data_loader.load_data([0, 1])
    #data_loader.normalize_train_data()  # 在本例中不需要归一化
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_4, epoch=100):
    batch_size = 16
    lr = 0.5
    W = np.zeros((1,1))
    B = np.zeros((1,1))
    nn = NeuralNet_4(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=5)
    return nn, training_history

# n1 - 正类样本数量，n2 - 负类样本数量
def show_result(X, Y, n1, n2, w, b, x):
    x1 = X[Y==1] 
    x2 = X[Y==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[0:n1], [0]*n1, c='r', marker='x', label='学区房')
    plt.scatter(x2[0:n2], [0]*n2, c='b', marker='o', label='普通房')
    plt.grid()
    plt.legend(loc='upper right')
    # 画出分界点
    if x is not None:
        plt.scatter(x, 0, marker="*")
    # 画出分界线
    if w is not None:
        minv = np.min(X)
        maxv = np.max(X)
        plt.plot([minv,maxv],[minv*w+b,maxv*w+b])

    plt.show()

if __name__ == '__main__':
    epoch = 100
    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history = train(data_loader, epoch)
    iteration, val_loss, W, B = training_history.get_best()
    print("权重值 W =", W)
    print("偏置值 B =", B)
    weight = W[0, 0]
    bias = B[0, 0]
    x = - bias / weight
    print("安居房和商品房的单价的分界点为", x)
    training_history.show_loss()
    X, Y = data_loader.get_train()
    show_result(X, Y, 20, 30, weight, bias, x)

import os
from common.NeuralNet_5 import NeuralNet_5
from common.DataLoader_5 import DataLoader_5
from H5_1_ShowData import show_data, load_data
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_5 import TrainingHistory_5

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train5.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_5(file_path)
    data_loader.load_data([0, 1, 3]) # x,y, 学区房标签
    # data_loader.normalize_train_data()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_5):
    batch_size = 10
    epoch = 50
    lr = 1 # 0.01,0.1,0.5,1,10,50
    W = np.zeros((2,1))
    B = np.zeros((1,1))
    nn = NeuralNet_5(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=80)
    return nn, training_history

def show_result(W, B):
    show_data(100, 300, W, B)

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
    show_result(W, B)

    pred_x = np.array([[0.2, 0.8], [0.4, 0.9], [0.6, 0.5]])
    pred_y = nn.predict(pred_x, False)
    for i in range(pred_x.shape[0]):
        if pred_y[i] > 0.5:
            print("地理位置在", pred_x[i], "的房子分类预测值为",pred_y[i]," > 0.5, 是学区房。")
        else:
            print("地理位置在", pred_x[i], "的房子分类预测值为",pred_y[i]," < 0.5, 不是学区房。")

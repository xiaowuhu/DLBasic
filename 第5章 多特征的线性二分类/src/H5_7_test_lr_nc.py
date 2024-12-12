import os
from common.NeuralNet_5_7 import NeuralNet_5_7
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

def train(data_loader: DataLoader_5, lrt):
    batch_size = 10
    epoch = 50
    lr = 1e-5
    W = np.zeros((2,1))
    B = np.zeros((1,1))
    nn = NeuralNet_5_7(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, lrt, checkpoint=80)
    return nn, training_history

def show_result(W, B):
    show_data(100, 300, W, B)

lr_dict={
    0.01: 0.01,
    0.1: 0.1,
    0.9: 0
}

# 根据当前的学习率设置迭代次数和调整增长步长
class lr_tunner(object):
    def __init__(self, lr_dict, update_step=10):
        self.update_step = update_step
        self.iteration = 0
        self.lr_list = []
        lrs = list(lr_dict.keys())
        # 把学习率按照字典的值铺开成列表
        for i, start_lr in enumerate(lrs):
            if lr_dict[start_lr] == 0:
                break
            lr = start_lr
            while lr < lrs[i+1]:
                self.lr_list.append(lr)
                lr += lr_dict[start_lr]
        self.idx = 0
        self.current_lr = self.lr_list[self.idx]

    def step(self):
        self.iteration += 1
        if self.iteration == self.update_step:
            self.iteration = 0
            if self.idx < len(self.lr_list)-1:
                self.idx += 1
            self.current_lr = self.lr_list[self.idx]
        return self.current_lr


if __name__ == '__main__':
    lrt = lr_tunner(lr_dict, update_step=1000)
    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history = train(data_loader, lrt)
    training_history.show_loss()


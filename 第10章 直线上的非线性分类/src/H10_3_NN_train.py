import os
import sys
import math
import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_10 import DataLoader_10
from common.TrainingHistory import TrainingHistory
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import tpn2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 自定义的模型
class NN(Module):
    def __init__(self):
        # 初始化 forward 中需要的各自operator
        self.linear1 = layer.Linear(1, 2)  # 线性层1（输入层）
        self.tanh = activator.Tanh()       # 激活层
        self.linear2 = layer.Linear(2, 1)  # 线性层2（输出层）
        self.classifier = layer.Logistic()# 二分类
        self.loss = loss.CrossEntropy2()   # 二分类交叉熵损失函数
        self.lce = layer.LogisticCrossEntropy() # 二分类函数+交叉熵损失函数

    def forward(self, X):
        Z = self.linear1.forward(X)
        A = self.tanh.forward(Z)
        Z = self.linear2.forward(A)
        A = self.classifier.forward(Z)
        # 这里不需要每次都计算损失函数值，只在需要的时候计算
        return A
    
    def backward(self, predict, label):
        delta = self.lce.backward(predict, label) # 代替下面两行
        # delta = self.loss.backward(predict, label)
        # delta = self.classifier.backward(delta)
        delta = self.linear2.backward(delta)
        delta = self.tanh.backward(delta)
        self.linear1.backward(delta)

    def update(self, lr):
        self.linear1.update(lr)
        self.linear2.update(lr)

    def predict(self, X):
        return self.forward(X)

    def save(self, name):
        super().save_parameters(name, (self.linear1, self.linear2))

    def load(self, name):
        super().load_parameters(name, (self.linear1, self.linear2))

def load_data():
    file_name = "train10.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_10(file_path)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader


# 计算损失函数和准确率
def check_loss(data_loader, model, training_history:TrainingHistory, iteration:int):
    # 训练集
    x, label = data_loader.get_train()
    predict = model.forward(x)
    train_loss = model.loss.forward(predict, label)
    train_accu = tpn2(predict, label)
    # 验证集
    x, label = data_loader.get_val()
    predict = model.forward(x)
    val_loss = model.loss.forward(predict, label)
    val_accu = tpn2(predict, label)
    # 记录历史
    
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_10, 
        model: Module,
        params: HyperParameters,
        checkpoint = 1,
):
    training_history = TrainingHistory()
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    check_iteration = int(batch_per_epoch * checkpoint)    
    iteration = 0 # 每一批样本算作一次迭代
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(params.batch_size, batch_id)
            batch_predict = model.forward(batch_X)
            model.backward(batch_predict, batch_label)
            model.update(params.learning_rate)
            iteration += 1
            if iteration % check_iteration == 0:
                train_loss, train_accu, val_loss, val_accu = check_loss(
                    data_loader, model, training_history, iteration)
                print("轮数 %d, 迭代 %d, 训练集: loss %f, accu %f, 验证集: loss %f, accu %f, 学习率:%.4f" \
                       %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, params.learning_rate))
    return training_history

if __name__=="__main__":
    data_loader = load_data()
    params = HyperParameters(max_epoch=500, batch_size=32, learning_rate=0.1)
    model = NN()
    training_history = train_model(data_loader, model, params, checkpoint=10)
    #training_history.save_history("train_history_10_3.txt")
    # model.save("model_10_3")
    training_history.show_loss()

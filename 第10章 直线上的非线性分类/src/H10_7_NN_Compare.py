import os
import sys
import math
import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_10 import DataLoader_10
from common.TrainingHistory import TrainingHistory
from common.Module import Module, Sequential
from common.HyperParameters import HyperParameters
from common.Estimators import tpn2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


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
    train_loss = model.compute_loss(predict, label)
    train_accu = tpn2(predict, label)
    # 验证集
    x, label = data_loader.get_val()
    predict = model.forward(x)
    val_loss = model.compute_loss(predict, label)
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
                print("轮数 %d, 迭代 %d, 训练集: loss %.6f, accu %.4f, 验证集: loss %.6f, accu %.4f, 学习率:%.4f" \
                       %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, params.learning_rate))
    return training_history





if __name__=="__main__":

    model = Sequential(
        layer.Linear(1, 2, init_method="xavier"),
        activator.Tanh(),
        layer.Linear(2, 1, init_method="xavier"),
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    data_loader = load_data()
    params = HyperParameters(max_epoch=300, batch_size=32, learning_rate=0.1)
    training_history = train_model(data_loader, model, params, checkpoint=10)
    #training_history.save_history("train_history_sgd.txt")

    model = Sequential(
        layer.Linear(1, 2, init_method="xavier", optimizer="AdaGrad"),
        activator.Tanh(),
        layer.Linear(2, 1, init_method="xavier", optimizer="AdaGrad"),
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    data_loader = load_data()
    params = HyperParameters(max_epoch=300, batch_size=32, learning_rate=0.1)
    training_history = train_model(data_loader, model, params, checkpoint=10)
    #training_history.save_history("train_history_adagrad.txt")
    
    model = Sequential(
        layer.Linear(1, 2, init_method="xavier", optimizer=("RMSProp", 0.7)),
        activator.Tanh(),
        layer.Linear(2, 1, init_method="xavier", optimizer=("RMSProp",0.7)),
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    data_loader = load_data()
    params = HyperParameters(max_epoch=300, batch_size=32, learning_rate=0.1)
    training_history = train_model(data_loader, model, params, checkpoint=10)
    #training_history.save_history("train_history_rmsprop.txt")

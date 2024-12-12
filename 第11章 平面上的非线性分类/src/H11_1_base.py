import os
import math

from common.DataLoader_11 import DataLoader_11
from common.TrainingHistory_11 import TrainingHistory_11
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.Estimators import tpn2, tpn3, r2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "data", file_name)
    data_loader = DataLoader_11(file_path)
    data_loader.load_data()
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

# 计算损失函数和准确率
def check_loss(data_loader, model: Sequential, training_history:TrainingHistory_11, epoch:int, iteration:int, learning_rate:float):
    # 训练集
    x, label = data_loader.get_train()
    predict = model.forward(x)
    train_loss = model.compute_loss(predict, label)
    if model.net_type == "Regression":
        train_accu = r2(label, train_loss)
    elif model.net_type == "BinaryClassifier":
        train_accu = tpn2(predict, label)
    elif model.net_type == "Classifier":
        train_accu = tpn3(predict, label)
    # 验证集
    x, label = data_loader.get_val()
    predict = model.forward(x)
    val_loss = model.compute_loss(predict, label)
    if model.net_type == "Regression":
        val_accu = r2(label, val_loss)
    elif model.net_type == "BinaryClassifier":
        val_accu = tpn2(predict, label)
    elif model.net_type == "Classifier":
        val_accu = tpn3(predict, label)

    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)

    print("轮数 %d, 迭代 %d, 训练集: loss %.6f, accu %.4f, 验证集: loss %.6f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_11, 
        model: Sequential,
        params: HyperParameters,
        checkpoint = 1,
):
    training_history = TrainingHistory_11()
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
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader, model, training_history, epoch, iteration, params.learning_rate)
    check_loss(data_loader, model, training_history, epoch, iteration, params.learning_rate)
    return training_history

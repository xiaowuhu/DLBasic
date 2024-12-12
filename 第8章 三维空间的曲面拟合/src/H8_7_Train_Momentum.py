import os
import sys
import math
import numpy as np
import common.Layers_7 as layer
import common.Activators as activator
import common.LossFunctions as loss
import common.Optimizers as Optimizer
from common.DataLoader_8 import DataLoader_8
from common.TrainingHistory_8 import TrainingHistory_8
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import r2
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 自定义的模型
class NN(Module):
    def __init__(self, optimizer = "SGD"):
        self.linear1 = layer.Linear(2, 12, optimizer=optimizer)
        self.tanh = activator.Tanh()
        self.linear2 = layer.Linear(12, 1, optimizer=optimizer)
        self.loss = loss.MSE()

    def forward(self, X):
        Z1 = self.linear1.forward(X)
        A1 = self.tanh.forward(Z1)
        Z2 = self.linear2.forward(A1)
        return Z2
    
    # X:输入批量样本, Y:标签, Z:预测值
    def backward(self, predict, label):
        dZ2 = self.loss.backward(predict, label)
        dA1 = self.linear2.backward(dZ2)
        dZ1 = self.tanh.backward(dA1)
        self.linear1.backward(dZ1)

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
    file_name = "train8.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_8(file_path)
    data_loader.load_data()
    data_loader.StandardScaler_X()
    data_loader.StandardScaler_Y()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader


# 计算损失函数和准确率
def check_loss(
        data_loader: DataLoader_8, 
        model: NN, 
        training_history:TrainingHistory_8, 
        iteration:int
):
    # 训练集
    x, y = data_loader.get_train()
    z = model.forward(x)
    train_loss = model.loss.forward(z, y)
    train_accu = r2(y, train_loss)
    # 验证集
    x, y = data_loader.get_val()
    if x is not None:
        z = model.forward(x)
        val_loss = model.loss.forward(z, y)
        val_accu = r2(y, val_loss)
    else:
        val_accu = train_accu
        val_loss = train_loss
    # 记录历史
    
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_8, 
        model: NN,
        params: HyperParameters,
        checkpoint = None,
):
    training_history = TrainingHistory_8()
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
            if iteration == 1 or iteration % check_iteration == 0:
                train_loss, train_accu, val_loss, val_accu = check_loss(
                    data_loader, model, training_history, iteration)
                print("轮数 %d, 迭代 %d, 训练集: loss %f, accu %f, 验证集: loss %f, accu %f, 学习率:%.4f" \
                       %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, params.learning_rate))
    return training_history

def save_history(training_history: TrainingHistory_8, name):
    history = training_history.get_history()
    file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
    np.savetxt(file_path, history, fmt='%.6f')


if __name__=="__main__":
    data_loader = load_data()
    params = HyperParameters(max_epoch=500, batch_size=32, learning_rate=0.1)
    model = NN("Momentum")
    history_momentum = train_model(data_loader, model, params, checkpoint=10)
#    save_history(history_momentum, "history_momentum_8_7.txt")
    model = NN("SGD")
    history_sgd = train_model(data_loader, model, params, checkpoint=10)
#    save_history(history_sgd, "history_sgd_8_7.txt")


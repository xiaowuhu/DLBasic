import os
import math
import numpy as np

from common.DataLoader_13 import DataLoader_13
import common.Layers as layer
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
from common.Estimators import r2, tpn2, tpn3
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_13(train_file_path, test_file_path)
    data_loader.load_data()
    data_loader.to_onehot(8)
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

# 计算损失函数和准确率
def check_loss(
        data_loader:DataLoader_13, 
        batch_size: int, batch_id: int, 
        model: Sequential, 
        training_history:TrainingHistory, 
        epoch:int, iteration:int, 
        learning_rate:float
):
    # 训练集
    x, label = data_loader.get_batch(batch_size, batch_id)
    train_loss, train_accu = model.compute_loss_accuracy(x, label)
    # 验证集
    x, label = data_loader.get_val()
    val_loss, val_accu = model.compute_loss_accuracy(x, label)
    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    print("轮数 %d, 迭代 %d, 训练集: loss %.6f, accu %.4f, 验证集: loss %.6f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_13, 
        model: Sequential,
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
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
    return training_history

def test_model(data_loader: DataLoader_13, model: Sequential):
    test_x, label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))
    data = model.operator_seq[0].forward(x[[0,50,100,150,200,250,300,350]])
    print(data)
    for i in range(8):
        plt.plot((0,data[i,0]),(0,data[i,1]), marker='o', label=str(i))
    plt.grid()
    plt.legend()
    plt.show()

def build_model():
    model = Sequential(
        layer.Linear(5, 2, init_method="normal", optimizer="Adam"),
        layer.Linear(2, 8, init_method="normal", optimizer="Adam"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy())
    return model


def draw_linear2_weights(model):
    data = model.operator_seq[1].WB.W.T
    for i in range(8):
        plt.plot((0,data[i,0]),(0,data[i,1]), marker='o', label=str(i))
    plt.grid()
    plt.legend()
    plt.show()


if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train13.txt", "test13.txt")
    params = HyperParameters(max_epoch=200, batch_size=32, learning_rate=0.1)
    #training_history = train_model(data_loader, model, params, checkpoint=1)
    #training_history.show_loss()
    # model.save("model_13_2_two_linear")
    # # test
    model.load("model_13_2_two_linear")
    test_model(data_loader, model)
    draw_linear2_weights(model)

import math
import common.Layers as layer
import common.Activators as activator

from common.DataLoader_11 import DataLoader_11
from common.TrainingHistory_11 import TrainingHistory_11
from common.Module import Sequential
from common.HyperParameters import HyperParameters
import matplotlib.pyplot as plt

from H11_1_base import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def train_model(
        data_loader: DataLoader_11, 
        model: Sequential,
        params: HyperParameters,
        checkpoint = None,
):
    training_history = TrainingHistory_11()
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    if checkpoint is None:
        checkpoint = batch_per_epoch # 每个epoch记录一次
    iteration = 0 # 每一批样本算作一次迭代
    for epoch in range(params.max_epoch):
        if epoch + 1 in [10, 30, 100]:
            model.save("model_11_moon_" + str(epoch+1))        
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(params.batch_size, batch_id)
            batch_predict = model.forward(batch_X)
            model.backward(batch_predict, batch_label)
            model.update(params.learning_rate)
            iteration += 1
            if iteration % checkpoint == 0:
                check_loss(data_loader, model, training_history, epoch, iteration, params.learning_rate)
    check_loss(data_loader, model, training_history, epoch, iteration, params.learning_rate)
    return training_history

def build_model():
    model = Sequential(
        layer.Linear(2, 2),
        activator.Tanh(),     # 激活层
        layer.Linear(2, 1),   # 线性层2（输出层）
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    return model

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train11-moon.txt")
    params = HyperParameters(max_epoch=100, batch_size=32, learning_rate=0.1)
    training_history = train_model(data_loader, model, params)
    training_history.show_loss()

import os
import math

from common.DataLoader_12 import DataLoader_12
import common.Layers as layer
import common.Activators as activator
from common.Module_4 import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    data_loader = DataLoader_12(file_path)
    data_loader.load_data()
    data_loader.to_onehot(3)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

# 计算损失函数和准确率
def check_loss(
        data_loader:DataLoader_12, 
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
        data_loader: DataLoader_12, 
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
            model.backward_last(batch_predict, batch_label)
            model.update(params.learning_rate)
            iteration += 1
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
    return training_history

def test_model(data_loader: DataLoader_12, model: Sequential):
    test_x, label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))

def build_model():
    model1 = Sequential(
        layer.Linear(3, 1, init_method="xavier", optimizer="SGD"),
        activator.Tanh(),
        name = "model1"
    )
    model2 = Sequential(
        layer.Linear(3, 1, init_method="xavier", optimizer="SGD"),
        activator.Tanh(),
        name = "model2"
    )
    model3 = Sequential(
        layer.Linear(3, 1, init_method="xavier", optimizer="SGD"),
        activator.Tanh(),
        name = "model3"
    )
    concat = layer.Concat((model1, model2, model3), (3,3,3), (1,1,1))
    model = Sequential(
        concat,
        layer.Linear(3, 3, init_method="kaiming", optimizer="SGD"),
        name = "model"
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model


if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train12_4.txt")
    params = HyperParameters(max_epoch=1000, batch_size=32, learning_rate=0.01)
    training_history = train_model(data_loader, model, params, checkpoint=5)
    training_history.show_loss()
    # model.save("model_12_4_Toy")

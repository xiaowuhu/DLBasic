import os
import math
import common.Layers_5 as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_9 import DataLoader_9
from common.TrainingHistory_9 import TrainingHistory_9
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import r2
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 自定义的模型
class NN(Module):
    def __init__(self, init_method = "normal", optimizer = "SGD", regularizer = None):
        # 初始化 forward 中需要的operator
        num_hidden_1 = 16
        num_hidden_2 = 20
        num_hidden_3 = 16

        self.linear1 = layer.Linear(1, num_hidden_1, init_method=init_method, optimizer=optimizer, regularizer=regularizer)
        self.tanh1 = activator.Tanh()

        self.linear2 = layer.Linear(num_hidden_1, num_hidden_2, init_method=init_method, optimizer=optimizer, regularizer=regularizer)
        self.tanh2 = activator.Tanh()

        self.linear3 = layer.Linear(num_hidden_2, num_hidden_3, init_method=init_method, optimizer=optimizer, regularizer=regularizer)
        self.tanh3 = activator.Tanh()

        self.linear4 = layer.Linear(num_hidden_3, 1, init_method=init_method, optimizer=optimizer, regularizer=regularizer)
        self.loss = loss.MSE()

    def forward(self, X):
        Z = self.linear1.forward(X)
        A = self.tanh1.forward(Z)

        Z = self.linear2.forward(A)
        A = self.tanh2.forward(Z)

        Z = self.linear3.forward(A)
        A = self.tanh3.forward(Z)

        Z = self.linear4.forward(A)
        return Z
    
    # X:输入批量样本, Y:标签, Z:预测值
    def backward(self, predict, label):
        delta = self.loss.backward(predict, label)
        delta = self.linear4.backward(delta)
        
        delta = self.tanh3.backward(delta)
        delta = self.linear3.backward(delta)   

        delta = self.tanh2.backward(delta)
        delta = self.linear2.backward(delta)

        delta = self.tanh1.backward(delta)
        self.linear1.backward(delta)

    def update(self, lr):
        self.linear1.update(lr)
        self.linear2.update(lr)
        self.linear3.update(lr)
        self.linear4.update(lr)

    def get_regular_cost(self):
        regular_cost = 0
        regular_cost += self.linear1.get_regular_cost()
        regular_cost += self.linear2.get_regular_cost()
        regular_cost += self.linear3.get_regular_cost()
        regular_cost += self.linear4.get_regular_cost()
        return regular_cost

    def save(self, name):
        super().save_parameters(
            name, 
            (self.linear1, self.linear2, self.linear3, self.linear4)
        )

    def load(self, name):
        super().load_parameters(
            name, 
            (self.linear1, self.linear2, self.linear3, self.linear4)
        )

def load_data():
    file_name = "train9.txt"
    file_path_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    file_name = "val9.txt"
    file_path_val = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    data_loader = DataLoader_9(file_path_train, file_path_val)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    data_loader.MinMaxScaler_val_XY()
    return data_loader


# 计算损失函数和准确率
def check_loss(
        data_loader: DataLoader_9, 
        model: NN, 
        training_history:TrainingHistory_9, 
        iteration:int
):
    # 训练集
    x, y = data_loader.get_train()
    z = model.forward(x)
    train_loss = model.loss.forward(z, y)
    regular_loss = model.get_regular_cost()
    train_loss += regular_loss / x.shape[0]
    train_accu = r2(y, train_loss)
    # 验证集
    x, y = data_loader.get_val()
    if x is not None:
        z = model.forward(x)
        val_loss = model.loss.forward(z, y) + regular_loss / x.shape[0]
        val_accu = r2(y, val_loss)
    else:
        val_accu = train_accu
        val_loss = train_loss
    # 记录历史
    
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_9, 
        model: Module,
        params: HyperParameters,
        checkpoint = 1,
):
    training_history = TrainingHistory_9()
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
    params = HyperParameters(max_epoch=100000, batch_size=32, learning_rate=0.002)
    model = NN(init_method="xavier", optimizer="Momentum", regularizer=("L1", 0.01))
    training_history = train_model(data_loader, model, params, checkpoint=100)
    model.save("L1_model")
    training_history.save_history("training_history_9_5_L1.txt")

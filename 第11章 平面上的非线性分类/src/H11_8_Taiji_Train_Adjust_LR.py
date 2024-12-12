import numpy as np
import common.Layers as layer
import common.Activators as activator
from common.Module import Sequential
from common.HyperParameters import HyperParameters
import matplotlib.pyplot as plt
from H11_1_base import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "data", file_name)
    data_loader = DataLoader_11(file_path)
    data_loader.load_data()
    data_loader.to_onehot(4)
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def build_model():
    model = Sequential(
        layer.Linear(2, 12, init_method="xavier", optimizer="Adam"),
        activator.Tanh(),     # 激活层
        layer.Linear(12, 4, init_method="xavier", optimizer="Adam"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model


def increase_lr(params: HyperParameters):
    params.learning_rate += 0.001
    return

    if params.learning_rate < 0.01:
        params.learning_rate += 0.001
        return
    
    if params.learning_rate < 0.1:
        params.learning_rate += 0.001
        return
    
def train_model(
        data_loader: DataLoader_11, 
        model: Sequential,
        params: HyperParameters,
):
    history = []
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    iteration = 0 # 每一批样本算作一次迭代
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(params.batch_size, batch_id)
            batch_predict = model.forward(batch_X)
            model.backward(batch_predict, batch_label)
            model.update(params.learning_rate)
            iteration += 1
            train_loss = check_loss(data_loader, model, epoch, iteration, params.learning_rate)
            history.append((params.learning_rate, train_loss))
            increase_lr(params)
    return history


# 计算损失函数和准确率
def check_loss(data_loader, model: Sequential, epoch:int, iteration:int, learning_rate:float):
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

    print("轮数 %d, 迭代 %d, 训练集: loss %.6f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, learning_rate))
    return train_loss

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train11-taiji.txt")
    params = HyperParameters(max_epoch=2, batch_size=4, learning_rate=0.001)
    history = train_model(data_loader, model, params)
    history = np.array(history)
    plt.plot(history[:,0], history[:,1])
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.grid()
    plt.show()


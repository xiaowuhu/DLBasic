import os
import math
import time
from common.DataLoader_12 import DataLoader_12
import common.Layers_7 as layer
import common.Activators as activator
from common.Module_7 import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_minist_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    data_loader.to_onehot(10)
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
    print("轮数 %d, 迭代 %d, 训练集: loss %.4f, accu %.4f, 验证集: loss %.4f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    return train_loss, train_accu, val_loss, val_accu

def train_model(
        data_loader: DataLoader_12, 
        model: Sequential,
        params: HyperParameters,
        lrs: LRScheduler,
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
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
    return training_history

def build_model():
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

# 在网络后端的BN
def build_bn1_model():
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        layer.BatchNorm1d(32),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

# 在网络前端的BN
def build_bn2_model():
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        layer.BatchNorm1d(64),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

# 在激活函数后面的BN
def build_bn3_model():
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.BatchNorm1d(32),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

# 网络前端后端都有BN
def build_bn4_model():
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        layer.BatchNorm1d(64),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        layer.BatchNorm1d(32),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

# 比较正常的网络与有 BN 的网络
def test1():
    models = []
    models.append(build_model())
    models.append(build_bn1_model())

    data_loader = load_minist_data()
    lrs = LRScheduler.step_lrs(0.2, 0.9, 5000)
    #lrs = LRScheduler.fixed_lrs(0.1)
    names = ["w/o bn", "bn"]
    ls = ["-", ":"]
    params = HyperParameters(max_epoch=5, batch_size=64)
    for i in range(len(models)):
        print(names[i])
        time1 = time.time()
        training_history = train_model(data_loader, models[i], params, lrs, checkpoint=0.1)
        models[i].save("model_12_7_bn")
        time2 = time.time()
        print(time2 - time1)
        history = training_history.get_history()
        start = 1
        iter, val_loss = history[start:, 0], history[start:,3]
        val_loss = training_history.moving_average(val_loss, 5)
        plt.plot(iter[start:start+len(val_loss)], val_loss, label=names[i], linestyle=ls[i])
    plt.legend()
    plt.grid()
    plt.show()

# 比较不同位置 BN 
def test2():
    models = []
    models.append(build_bn1_model())
    models.append(build_bn2_model())
    models.append(build_bn3_model())
    models.append(build_bn4_model())

    data_loader = load_minist_data()
    lrs = LRScheduler.step_lrs(0.2, 0.9, 5000)
    #lrs = LRScheduler.fixed_lrs(0.1)
    names = ["在网络后端的BN", "在网络前端的BN", "在激活函数后面的BN", "网络前端后端都有BN"]
    ls = ["-", ":", "-.", "--"]
    params = HyperParameters(max_epoch=5, batch_size=64)
    for i in range(len(models)):
        print(names[i])
        time1 = time.time()
        training_history = train_model(data_loader, models[i], params, lrs, checkpoint=0.1)
        time2 = time.time()
        print(time2 - time1)
        history = training_history.get_history()
        start = 1
        iter, val_loss = history[start:, 0], history[start:,3]
        val_loss = training_history.moving_average(val_loss, 5)
        plt.plot(iter[start:start+len(val_loss)], val_loss, label=names[i], linestyle=ls[i])
    plt.legend()
    plt.grid()
    plt.show()

# 比较不同位置 BN 
def test3():
    models = []
    models.append(build_bn1_model())
    models.append(build_bn3_model())

    data_loader = load_minist_data()
    lrs = LRScheduler.step_lrs(0.1, 0.9, 5000)
    #lrs = LRScheduler.fixed_lrs(0.1)
    names = ["在激活函数前面的BN", "在激活函数后面的BN"]
    ls = ["-", ":"]
    params = HyperParameters(max_epoch=10, batch_size=64)
    for i in range(len(models)):
        print(names[i])
        time1 = time.time()
        training_history = train_model(data_loader, models[i], params, lrs, checkpoint=0.1)
        time2 = time.time()
        print(time2 - time1)
        history = training_history.get_history()
        start = 1
        iter, val_loss = history[start:, 0], history[start:,3]
        val_loss = training_history.moving_average(val_loss, 5)
        plt.plot(iter[start:start+len(val_loss)], val_loss, label=names[i], linestyle=ls[i])
    plt.legend()
    plt.grid()
    plt.show()


if __name__=="__main__":
    test1()
    test2()
    test3()

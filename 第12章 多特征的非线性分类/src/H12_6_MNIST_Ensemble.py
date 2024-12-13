import os
import numpy as np
import math

from common.DataLoader_12 import DataLoader_12
import common.Layers as layer
import common.Activators as activator
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
from common.Estimators import r2, tpn2, tpn3
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

from H12_6_MNIST_Train import build_model1, build_model3, build_model2

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def load_minist_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    #data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    return data_loader

def load_feature_data(name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    data_loader = DataLoader_12(file_path)
    data_loader.load_data()
    return data_loader


def test_model(data_loader: DataLoader_12, model: Sequential, name):
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("模型 %s - 测试集: loss %.4f, accu %.4f" %(name, test_loss, test_accu))


def inference(model: Sequential, X):
    result = model.forward_without_softmax(X)
    return result

# 用三个模型从训练集中提取没有做softmax之前的线性计算结果
def generate_feature():
    features_train = []
    features_test = []

    data_loader = load_minist_data()
    train_X, train_label = data_loader.get_train()
    test_X, test_label = data_loader.get_test()
    test_X = data_loader.StandardScaler_pred_X(test_X)

    model1 = build_model1()
    model1.load("model_12_6_SGD")
    features_train.append(inference(model1, train_X))
    features_test.append(inference(model1, test_X))

    model2 = build_model2()
    model2.load("model_12_6_Adam")
    features_train.append(inference(model2, train_X))
    features_test.append(inference(model2, test_X))

    model3 = build_model3()
    model3.load("model_12_6_Momentum")
    features_train.append(inference(model3, train_X))
    features_test.append(inference(model3, test_X))

    features_train = np.hstack(features_train)
    print(features_train.shape)
    data = np.hstack((features_train, train_label))
    print(data.shape)
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_path, "MNIST_features_train.txt")
    np.savetxt(file_name, data, fmt="%.6f")

    features_test = np.hstack(features_test)
    print(features_test.shape)
    data = np.hstack((features_test, test_label))
    print(data.shape)
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_path, "MNIST_features_test.txt")
    np.savetxt(file_name, data, fmt="%.6f")

# 集成模型
def build_model():
    model = Sequential(
        layer.Linear(30, 16, init_method="xavier", optimizer="Adam"),
        activator.Tanh(),
        layer.Linear(16, 10, init_method="xavier", optimizer="Adam"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model


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

def train_feature_model():
    data_loader = load_feature_data("MNIST_features_train.txt")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X()
    data_loader.split_data(0.9)
    lrs = LRScheduler.step_lrs(0.001, 0.99, 5000)
    params = HyperParameters(max_epoch=50, batch_size=32)
    model = build_model()
    training_history = train_model(data_loader, model, params, lrs, checkpoint=1)   
    training_history.show_loss() 
    #model.save("model_12_6_feature")
    return data_loader.x_mean, data_loader.x_std


def test_feature():
    data_loader = load_feature_data("MNIST_features_test.txt")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X()
    test_X, test_label = data_loader.get_train()

    model_feature = build_model()
    model_feature.load("model_12_6_feature")
    test_loss, test_accu = model_feature.compute_loss_accuracy(test_X, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))


if __name__=="__main__":
    generate_feature()  # 生成新的特征数据
    train_feature_model() # 用新特征数据训练集成模型
    test_feature()  # 测试

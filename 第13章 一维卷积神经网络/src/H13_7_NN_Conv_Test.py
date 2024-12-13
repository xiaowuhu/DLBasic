import os
import math
import numpy as np

from common.DataLoader_13 import DataLoader_13
import common.Layers as layer
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

from H13_7_NN_Conv import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_13(train_file_path, test_file_path)
    data_loader.load_data()
    data_loader.add_channel_info()
    data_loader.to_onehot(4)
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader


def test_model(data_loader: DataLoader_13, model: Sequential):
    test_x, label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))


def show_filter_shape(model):
    conv_layer = model.operator_seq[0]
    filter = conv_layer.WB.W
    print("filter shape:", filter.shape)
    print(filter)
    titles = ["$w_1$","$w_2$","$w_3$","$w_4$"]
    fig = plt.figure(figsize=(8,3))
    for i in range(filter.shape[0]):
        ax = fig.add_subplot(1, 4, i+1)
        ax.plot(filter[i].reshape(-1), marker='.')
        ax.grid()
        ax.set_title(titles[i])
        ax.set_ylim(-2.1,2.1)
        ax.set_aspect("equal")
    plt.show()

# 检查每类样本的卷积结果后的池化最大输出在哪个核上
def check_max_pool(data_loader: DataLoader_13, model: Sequential):
    test_x, label = data_loader.get_test()
    test_x = data_loader.StandardScaler_pred_X(test_x)
    ids = []
    for i in range(0, 400, 50): # 每50个样本是一类，一共400个样本
        z1 = model.operator_seq[0].forward(test_x[[i]]) # 卷积 Conv
        z2 = model.operator_seq[1].forward(z1) # 最大池化 MaxPool
        z3 = model.operator_seq[2].forward(z2) # Flatten
        z4 = model.operator_seq[3].forward(z3) # 线性分类 Linear
        #print("z1", z1.shape, z1)
        #print("z2", z2.shape, z2)
        print("---- 样本号:",i)
        #print("z1", z1.shape, z1)
        #print("z2", z2.shape, z2)
        #print(model.operator_seq[1].argmax.flatten())
        id = model.operator_seq[1].argmax[0,i//50%4][0]
        print("最大池化特征值序号:", id)
        ids.append(id)
        #print("z3", z3.shape, z3)
        #print("z4", z4.shape, z4)

    titles = ["sin","cos","sawtooth","flat","-sin","-cos","-sawtooth","-flat"]
    fig = plt.figure(figsize=(12,5))
    for i in range(len(ids)):
        id = ids[i]
        ax = fig.add_subplot(2, 4, i+1)
        sample = test_x[[i*50]].flatten() # 样本 i*50 
        x = np.linspace(0,4,5)
        ax.plot(x[id:id+3], sample[id:id+3]) # 绘制三个点及其之间的连线
        ax.grid()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(0,4)
        ax.set_title(titles[i] + "特征" + str(id))
    plt.show()

def search_kernal_prefered_shape(model):
    feature = np.linspace(-1,1,10)
    # 三个特征遍历
    X = np.zeros((100000,1,5))
    id = 0
    for i in feature:
        for j in feature:
            for k in feature:
                for m in feature:
                    for n in feature:
                        X[id] = np.array([i,j,k,m,n])
                        id += 1
    Z = model.forward(X)
    result_all = np.argmax(Z, axis=1)
    fig = plt.figure(figsize=(12,5))
    titles = ["sin", "cos", "sawtooth", "flat", "-sin", "-cos", "-sawtooth", "-flat"]
    for i in range(4):
        ax = fig.add_subplot(2, 4, i+1)
        X_i = X[result_all == i]  # 获得 i 类的所有样本
        Z_i = Z[result_all == i]  # 获得 i 类的分类值
        id1 = np.argmin(Z_i[:,i]) # 分类值最小的样本序号
        X_mean = np.mean(X_i, axis=0)  # 样本平均值
        id2 = np.argmax(Z_i[:,i])  # 分类值最大的样本序号
        ax.plot(range(5), X_i[id2].flatten(), marker="o", label="high")
        ax.plot(range(5), X_mean.flatten(), linestyle="--", label="mean")
        ax.plot(range(5), X_i[id1].flatten(), marker='.', label="low")
        ax.grid()
        ax.legend()
        ax.set_ylim(-1.1,1.1)
        ax.set_title(titles[i])
    plt.show()

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train13_4.txt", "test13_4.txt")
    model.load("model_13_7_conv_pool2")
    #test_model(data_loader, model)
    show_filter_shape(model)
    check_max_pool(data_loader, model)
    #search_kernal_prefered_shape(model)
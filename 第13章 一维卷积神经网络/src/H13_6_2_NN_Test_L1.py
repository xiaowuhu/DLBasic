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

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=10)

def load_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_13(train_file_path, test_file_path)
    data_loader.load_data()
    data_loader.add_channel_info()
    data_loader.to_onehot(8)
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader


def test_class_data(x, model):
    result = model.operator_seq[0].forward(x)
    result = np.mean(result, axis=0)
    return result

def get_weights(model: Sequential):
    WB = model.operator_seq[2].get_parameters()
    return WB.W

def test_model(data_loader: DataLoader_13, model: Sequential):
    test_x, label = data_loader.get_test()
    test_x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(test_x, label)
    print("测试集: loss %.6f, accu %.4f" %(test_loss, test_accu))
    
    W = get_weights(model)

    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(8):
        x1 = test_x[i*50:i*50+50]
        result = test_class_data(x1, model)
        if i+1 in [1,4,5,8]:
            # z1, z2
            x, y = result[0,0], result[0,1]
            ax1.plot((0, x),(0, y), marker='.',label=str(i+1))
            ax1.text(x, y, "$c_"+str(i+1)+"$")
            x, y = W[0,i], W[1,i]
            ax1.plot((0,W[0,i]),(0,W[1,i]), marker='^', label=str(i+1))
            ax1.text(x, y, "$w_"+str(i+1)+"$")

        else:
            # z3, z4
            x, y = result[1,0], result[1,1]
            ax2.plot((0, x),(0, y), marker='.',label=str(i+1))
            ax2.text(x, y, "$c_"+str(i+1)+"$")
            x, y = W[2,i], W[3,i]
            ax2.plot((0,x),(0,y), marker='^', label=str(i+1))
            ax2.text(x, y, "$w_"+str(i+1)+"$")

    ax1.grid()
    ax1.set_title("类别:1,4,5,8")
    ax2.grid()
    ax2.set_title("类别:2,3,6,7")
    plt.show()


def show_filter_shape(model):
    conv_layer = model.operator_seq[0]
    filter = conv_layer.WB.W
    print("filter shape:", filter.shape)
    print(filter)
    fig = plt.figure(figsize=(6,3))
    for i in range(filter.shape[0]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.plot(filter[i].reshape(-1), marker='.')
        ax.set_ylim((-3,3))
        ax.grid()
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
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1)
        X_i = X[result_all == i]
        Z_i = Z[result_all == i]
        id1 = np.argmin(Z_i[:,i])
        X_mean = np.mean(X_i, axis=0)
        id2 = np.argmax(Z_i[:,i])
        ax.plot(range(5), X_i[id2].flatten(), marker="o", label="high")
        ax.plot(range(5), X_mean.flatten(), linestyle="--", label="mean")
        ax.plot(range(5), X_i[id1].flatten(), marker='.', label="low")
        ax.grid()
        ax.legend()
        ax.set_ylim(-1.1,1.1)
        ax.set_title(titles[i])
    plt.show()

def build_model():
    input_channel = 1
    output_channel = 2
    kernel_length = 3
    input_length = 5
    stride = 2
    padding = 0
    output_length = 1 + (input_length + 2 * padding - kernel_length) // stride

    model = Sequential(
        layer.Conv1d((input_channel, input_length), (output_channel, kernel_length), 
                      stride=stride, padding=padding, init_method="normal", optimizer="Adam"),
        layer.Flatten((output_channel, output_length), output_length * output_channel),
        layer.Linear(output_length * output_channel, 8, init_method="normal", optimizer="Adam"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy())
    return model


if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train13.txt", "test13.txt")
    model.load("model_13_6_conv_L1")
    test_model(data_loader, model)
    show_filter_shape(model)   
    search_kernal_prefered_shape(model)


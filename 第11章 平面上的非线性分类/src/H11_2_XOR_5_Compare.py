import numpy as np
import common.Layers as layer
import common.Activators as activator

from common.DataLoader_11 import DataLoader_11
from common.Module import Sequential
import matplotlib.pyplot as plt

from H11_2_XOR_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def show_result(ax, model: Sequential, data_loader: DataLoader_11, num_hidden):
    count = 100
    # 原始样本点
    X, Y = data_loader.get_train()
    X = data_loader.de_StandardScaler_X(X)
    X0 = X[Y[:,0]==0]
    X1 = X[Y[:,0]==1]
    # 计算图片渲染元素
    x = np.linspace(-5, 5, count)
    y = np.linspace(-5, 5, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    pred_x = data_loader.StandardScaler_pred_X(input)
    output = model.predict(pred_x)
    Z = output.reshape(count, count)

    ax.contourf(X, Y, Z, cmap=plt.cm.Pastel1)
    ax.scatter(X0[:,0], X0[:,1], marker='o', label='0')
    ax.scatter(X1[:,0], X1[:,1], marker='x', label='1')
    ax.legend()
    ax.grid()
    ax.set_aspect(1)
    ax.set_title("隐层神经元数:"+str(num_hidden))

def build_model(num_hidden):
    model = Sequential(
        layer.Linear(2, num_hidden, init_method="xavier", optimizer="Momentum"),
        activator.Relu(),     # 激活层
        layer.Linear(num_hidden, 1, init_method="xavier", optimizer="Momentum"),   # 线性层2（输出层）
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    return model

if __name__=="__main__":
    num_hidden_list = [2, 3, 4, 8]
    # for i in range(len(num_hidden_list)):
    #     model = build_model(num_hidden_list[i])
    #     data_loader = load_data("train11-xor.txt")
    #     params = HyperParameters(max_epoch=500, batch_size=32, learning_rate=0.1)
    #     training_history = train_model(data_loader, model, params)
    #     training_history.show_loss()
    #     model.save("model_11_xor_"+str(num_hidden_list[i]))

    fig = plt.figure(figsize=(11, 3))
    for i in range(len(num_hidden_list)):
        model = build_model(num_hidden_list[i])
        model.load("model_11_xor_"+str(num_hidden_list[i]))
        data_loader = load_data("train11-xor.txt")
        ax = fig.add_subplot(1, 4, i+1)
        show_result(ax, model, data_loader, num_hidden_list[i])
    plt.show()

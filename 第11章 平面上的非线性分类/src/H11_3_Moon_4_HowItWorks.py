import numpy as np

from common.DataLoader_11 import DataLoader_11
import matplotlib.pyplot as plt
from H11_3_Moon_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def show_result(model: Sequential, data_loader: DataLoader_11):
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

    plt.figure(figsize=(5, 5))
    plt.contourf(X, Y, Z, cmap=plt.cm.Pastel1)

    plt.scatter(X0[:,0], X0[:,1], marker='o', label='0')
    plt.scatter(X1[:,0], X1[:,1], marker='x', label='1')
    plt.legend()
    plt.grid()
    plt.show()

def show_working_process(model:Sequential, data_loader: DataLoader_11):
    count = 20
    X, Y = data_loader.get_train()
    #X = data_loader.de_StandardScaler_X(X)

    X0 = X[Y[:,0]==0]  # 负类
    Z01 = model.operator_seq[0].forward(X0)
    A01 = model.operator_seq[1].forward(Z01)
    Z02 = model.operator_seq[2].forward(A01)

    X1 = X[Y[:,0]==1]  # 正类
    Z11 = model.operator_seq[0].forward(X1)
    A11 = model.operator_seq[1].forward(Z11)
    Z12 = model.operator_seq[2].forward(A11)

    # 准备空间数据
    x = np.linspace(-2.5, 2.5, count)  # 使用归一化好的数据范围生成空间矩阵
    y = np.linspace(-2.5, 2.5,count)   # 否则后面画网格时会有问题
    X, Y = np.meshgrid(x, y)
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    #pred_X = data_loader.StandardScaler_pred_X(input)
    Z1 = model.operator_seq[0].forward(input)
    A1 = model.operator_seq[1].forward(Z1)
    Z2 = model.operator_seq[2].forward(A1)
    A2 = model.classifier_function.forward(Z2)

    fig = plt.figure(figsize=(11,4))

    Z = Z1.reshape(count, count, 2)
    ax = fig.add_subplot(1, 3, 1)
    DrawGrid(ax, Z, count)
    ax.scatter(Z01[:,0], Z01[:,1], marker='.', label="0")
    ax.scatter(Z11[:,0], Z11[:,1], marker='x', label="1")
    ax.set_title("空间平移(第一层线性变换)")
    ax.set_aspect(1)

    ax = fig.add_subplot(1, 3, 2)
    A = A1.reshape(count, count, 2)
    DrawGrid(ax, A, count)
    ax.scatter(A01[:,0], A01[:,1], marker='.', label="0")
    ax.scatter(A11[:,0], A11[:,1], marker='x', label="1")
    ax.set_title("空间扭曲(第一层激活输出)")
    ax.set_aspect(1)

    ax = fig.add_subplot(1, 3, 3)
    Z = A2.reshape(count, count)
    ax.contourf(X, Y, Z, cmap=plt.cm.Pastel1)
    ax.scatter(X0[:,0], X0[:,1], marker='.', label="0")
    ax.scatter(X1[:,0], X1[:,1], marker='x', label="1")
    ax.set_title("分类结果(第二层概率输出)")
    ax.set_aspect(1)

    plt.show()

def DrawGrid(ax, Z, count):
    for i in range(count):
        for j in range(count):
            ax.plot(Z[:,j,0],Z[:,j,1],'-',c='gray',lw=0.1)
            ax.plot(Z[i,:,0],Z[i,:,1],'-',c='gray',lw=0.1)

if __name__=="__main__":
    data_loader = load_data("train11-moon.txt")
    X, Y = data_loader.get_val()
    model = build_model()
    # 展示最终结果
    model.load("model_11_moon_100")
    show_result(model, data_loader)
    # 展示训练过程
    model.load("model_11_moon_10")
    show_working_process(model, data_loader)
    model.load("model_11_moon_30")
    show_working_process(model, data_loader)
    model.load("model_11_moon_100")
    show_working_process(model, data_loader)


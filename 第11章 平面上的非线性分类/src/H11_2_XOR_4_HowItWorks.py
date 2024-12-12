import numpy as np

from common.DataLoader_11 import DataLoader_11
import matplotlib.pyplot as plt

from H11_2_XOR_3_Train import *

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
    X, Y = data_loader.get_train()
    X0 = X[Y[:,0]==0]  # 负类
    Z01 = model.operator_seq[0].forward(X0)
    A01 = model.operator_seq[1].forward(Z01)
    Z02 = model.operator_seq[2].forward(A01)

    X1 = X[Y[:,0]==1]  # 正类
    Z11 = model.operator_seq[0].forward(X1)
    A11 = model.operator_seq[1].forward(Z11)
    Z12 = model.operator_seq[2].forward(A11)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(X0[:,0], X0[:,1], marker='.', label='0')  # 负类
    ax.scatter(X1[:,0], X1[:,1], marker='x', label='1')  # 正类
    # 示踪点
    ax.scatter(X0[0,0], X0[0,1], marker='s', label='0', s=100)  # 负类
    ax.scatter(X1[0,0], X1[0,1], marker='^', label='1', s=100)  # 正类
    ax.grid()
    ax.legend()
    ax.set_title("1.原始数据")
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(Z01[:,0], Z01[:,1], marker='.', label='0')  # 负类
    ax.scatter(Z11[:,0], Z11[:,1], marker='x', label='1')  # 正类
    # 示踪点
    ax.scatter(Z01[0,0], Z01[0,1], marker='s', label='0', s=100)  # 负类
    ax.scatter(Z11[0,0], Z11[0,1], marker='^', label='1', s=100)  # 正类
    ax.grid()
    ax.legend()
    ax.set_title("2.第一层线性输出")
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(A01[:,0], A01[:,1], marker='.', label='0')  # 负类
    ax.scatter(A11[:,0], A11[:,1], marker='x', label='1')  # 正类
    # 示踪点
    ax.scatter(A01[0,0], A01[0,1], marker='s', label='0', s=100)  # 负类
    ax.scatter(A11[0,0], A11[0,1], marker='^', label='1', s=100)  # 正类
    ax.grid()
    ax.legend()
    ax.set_title("3.第一层激活输出")
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(Z02, [0] * Z02.shape[0], marker='.', label='0')  # 负类
    ax.scatter(Z12, [0] * Z12.shape[0], marker='x', label='1')  # 正类
    # 示踪点
    ax.scatter(Z02[0], 0, marker='s', label='0', s=100)  # 负类
    ax.scatter(Z12[0], 0, marker='^', label='1', s=100)  # 正类
    ax.grid()
    ax.legend()
    ax.set_title("4.第二层线性输出")
    plt.show()


if __name__=="__main__":
    data_loader = load_data("train11-xor.txt")
    model = build_model()
    model.load("model_11_xor_Relu")
    show_result(model, data_loader)
    show_working_process(model, data_loader)


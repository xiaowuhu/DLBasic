import os
from common.NeuralNet_6 import NeuralNet_6
from common.DataLoader_6 import DataLoader_6
import matplotlib.pyplot as plt
import numpy as np
from H6_4_NN_client import *


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def show_3d_result(net, count, x1, x2, x3):
    X1 = np.linspace(0,1,count)  
    X2 = np.linspace(0,1,count)
    X1, X2 = np.meshgrid(X1, X2)  # 建立 x1,x2 坐标网格，每个点都是一个 (x1,x2) 特征
    input = np.hstack((X1.reshape(count*count,1), X2.reshape(count*count,1))) # 平铺成两列
    A = net.forward(input) # 得到 softmax 预测结果
    # A 是 3 列 [0.01,0.02,0.97]，每一列代表一个类别的概率，取最大值显示[0.97]
    Z = np.max(A, axis=1).reshape(count,count)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='rainbow')

    ax = fig.add_subplot(122)
    ax.contourf(X1, X2, Z, cmap=plt.cm.Pastel1)
    ax.grid()
    # 绘制三类分类概率小于 0.95 的样本点
    ax.scatter(x1[:, 0], x1[:, 1], c='r', marker='+', label='0-武昌')
    ax.scatter(x2[:, 0], x2[:, 1], c='g', marker='.', label='1-汉口')
    ax.scatter(x3[:, 0], x3[:, 1], c='b', marker='*', label='2-汉阳')
    ax.legend(loc='lower right')

    plt.show()    

def load_result():
    file_name = "weights6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    W = np.loadtxt(file_path)
    file_name = "bias6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    B = np.loadtxt(file_path)[np.newaxis, :]
    return W, B

def load_nn(W, B):
    nn = NeuralNet_6(None, W, B)
    return nn

def get_sample_data(nn):
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2])
    X, Y = data_loader.get_train()
    y_pred = nn.predict(X, False)
    # 取出最大值类别
    y_class = np.max(y_pred, axis=1)
    # 获得边界处的点
    index = np.where(y_class < 0.95)
    x = X[index]
    y = Y[index]
    # 获得各自的类，作为整体，画点速度快
    x1 = x[y[:,0]==0]
    x2 = x[y[:,0]==1]
    x3 = x[y[:,0]==2]
    return x1, x2, x3

def load_data():
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2]) # 加载 横坐标，纵坐标，分类标签
    data_loader.split_data(0.8)
    return data_loader

def show_data_and_splitline(W, B):
    data_loader = load_data()
    X, Y = data_loader.get_val()

    x1 = X[Y[:, 0]==0]
    x2 = X[Y[:, 0]==1]
    x3 = X[Y[:, 0]==2]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='+')
    plt.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
    plt.scatter(x3[:, 0], x3[:, 1], c='b', marker='*')

    b = (B[0,1]-B[0,0]) / (W[1,0]-W[1,1])
    w = (W[0,1]-W[0,0]) / (W[1,0]-W[1,1])
    x = np.linspace(0, 1, 2)
    y = w * x + b
    plt.plot(x, y, c='b', linestyle="solid", label="1 vs 2")

    b = (B[0,0]-B[0,2]) / (W[1,2]-W[1,0])
    w = (W[0,0]-W[0,2]) / (W[1,2]-W[1,0])
    y = w * x + b
    plt.plot(x, y, c='g', linestyle="dotted", label="1 vs 3")

    b = (B[0,2]-B[0,1]) / (W[1,1]-W[1,2])
    w = (W[0,2]-W[0,1]) / (W[1,1]-W[1,2])
    y = w * x + b
    plt.plot(x, y, c='r', linestyle="dashed", label="2 vs 3")

    plt.xlim((0,1))
    plt.ylim((0,1))

    plt.grid()
    plt.legend()

    plt.show()

def show_confusion_matrix(nn):
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2]) # 加载 横坐标，纵坐标，分类标签
    X, Y = data_loader.get_train()
    y_pred = nn.predict(X, False)
    # 取出最大值类别
    y_pred = np.argmax(y_pred, axis=1, keepdims=True)
    # 手工计算混淆矩阵值
    confusion_matrix = np.zeros((3, 3))
    for i in range(y_pred.shape[0]):
        if y_pred[i] == Y[i]:
            confusion_matrix[Y[i], Y[i]] += 1
        else:
            confusion_matrix[Y[i], y_pred[i]] += 1
    print(confusion_matrix)
    plt.imshow(confusion_matrix, cmap="autumn_r")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, "%d"%(confusion_matrix[i, j]), ha='center', va='center')
    num_local = [0, 1, 2]
    plt.xticks(num_local)
    plt.yticks(num_local)
    plt.show()

if __name__ == '__main__':
    W, B = load_result()
    #show_data_and_splitline(W, B)
    
    # 准备数据
    print("加载数据...")
    print("权重值 w =", W)
    print("偏置值 b =", B)
    nn = load_nn(W, B)
    #x1, x2, x3 = get_sample_data(nn)
    #show_3d_result(nn, 100, x1, x2, x3)

    show_confusion_matrix(nn)

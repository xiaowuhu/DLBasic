import os
from common.DataLoader_6 import DataLoader_6
import matplotlib.pyplot as plt
import numpy as np
from H6_4_NN_client import *
from common.Functions_6 import softmax

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def load_result():
    file_name = "weights6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    W = np.loadtxt(file_path)
    file_name = "bias6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    B = np.loadtxt(file_path)[np.newaxis, :]
    return W, B


def show_X_Z_A(x1, x2, x3, Z, A):
    fig = plt.figure(figsize=(12,4))
    # 原始样本图
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(x1[:,0], x1[:,1], c='r', marker='+', label='0-武昌')
    ax.scatter(x2[:,0], x2[:,1], c='g', marker='.', label='1-汉口')
    ax.scatter(x3[:,0], x3[:,1], c='b', marker='*', label='2-汉阳')
    ax.legend()
    # 二维到三维的线性变换结果
    ax = fig.add_subplot(1, 3, 2,  projection='3d')
    ax.scatter(Z[0:50, 0], Z[0:50, 1], Z[0:50, 2], c='r', marker='+')
    ax.scatter(Z[50:100, 0], Z[50:100, 1], Z[50:100, 2], c='g', marker='.')
    ax.scatter(Z[100:150, 0], Z[100:150, 1], Z[100:150, 2], c='b', marker='*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 三维到三维的 softmax 变换结果
    ax = fig.add_subplot(1, 3, 3,  projection='3d')
    ax.scatter(A[0:50, 0], A[0:50, 1], A[0:50, 2], c='r', marker='+')
    ax.scatter(A[50:100, 0], A[50:100, 1], A[50:100, 2], c='g', marker='.')
    ax.scatter(A[100:150, 0], A[100:150, 1], A[100:150, 2], c='b', marker='*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def load_data():
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2]) # 加载 横坐标，纵坐标，分类标签
    data_loader.split_data(0.8)
    return data_loader

def show_data_and_w_vector():
    data_loader = load_data()
    X, Y = data_loader.get_val()

    W, B = load_result()
    print("权重值 w =", W)
    print("偏置值 b =", B)

    x1 = X[Y[:, 0]==0]
    x2 = X[Y[:, 0]==1]
    x3 = X[Y[:, 0]==2]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='+')
    plt.scatter(x2[:, 0], x2[:, 1], c='g', marker='.')
    plt.scatter(x3[:, 0], x3[:, 1], c='b', marker='*')

    plt.plot((0+0.5, W[0,0]+0.5), (0+0.5, W[1,0]+0.5), c='r', linestyle="solid", label="$w_1$", linewidth=2)
    plt.plot((0+0.5, W[0,1]+0.5), (0+0.5, W[1,1]+0.5), c='g', linestyle="dashed", label="$w_2$", linewidth=2)
    plt.plot((0+0.5, W[0,2]+0.5), (0+0.5, W[1,2]+0.5), c='b', linestyle="dotted", label="$w_3$", linewidth=2)

    plt.xlim((0,1))
    plt.ylim((0,1))

    plt.grid()
    plt.legend()

    plt.show()

if __name__ == '__main__':    
    W, B = load_result()
    print("权重值 w =", W)
    print("偏置值 b =", B)
    # 准备数据
    print("加载数据...")
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2])
    X, Y = data_loader.get_train()
    # 从三个类别中各取出 50 个样本
    x1 = X[Y[:, 0] == 0][:50]
    x2 = X[Y[:, 0] == 1][:50]
    x3 = X[Y[:, 0] == 2][:50]
    # 把三类样本合并
    X = np.vstack((x1, x2, x3))
    Z = np.dot(X, W) + B    
    A = softmax(Z)
    print("第0类线性变换结果:")
    print(Z[0:5])
    print("第1类线性变换结果:")
    print(Z[50:55])
    print("第2类线性变换结果:")
    print(Z[100:105])
    # 显示原始数据和 W 矢量
    show_data_and_w_vector()
    # 显示二维到三维的变换过程
    show_X_Z_A(x1, x2, x3, Z, A)

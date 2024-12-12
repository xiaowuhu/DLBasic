import os
import numpy as np
from common.DataLoader_11 import DataLoader_11
import matplotlib.pyplot as plt
from H11_4_Stone_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "data", file_name)
    data_loader = DataLoader_11(file_path)
    data_loader.load_data()
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def show_result(model, data_loader:DataLoader_11):
    count = 50
    X, Y = data_loader.get_train()
    X = data_loader.de_StandardScaler_X(X)
    # 原始样本点
    X0 = X[Y[:,0]==0]
    X1 = X[Y[:,0]==1]
    X2 = X[Y[:,0]==2]
    # 计算图片渲染元素
    x = np.linspace(-5,5,count)
    y = np.linspace(-5,5,count)
    X, Y = np.meshgrid(x, y)
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    pred_x = data_loader.StandardScaler_pred_X(input)
    output = model.predict(pred_x)
    Z = np.max(output, axis=1).reshape(count,count)

    plt.figure(figsize=(5, 5))
    plt.contourf(X, Y, Z, cmap=plt.cm.hot)

    plt.scatter(X0[:,0], X0[:,1], marker='s', label='0')
    plt.scatter(X1[:,0], X1[:,1], marker='o', label='1')
    plt.scatter(X2[:,0], X2[:,1], marker='^', label='2')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

def show_working_process(model:Sequential, data_loader: DataLoader_11):
    X, Y = data_loader.get_train()  # 已经标准化好的数据

    X0 = X[Y[:,0]==0]  # 0类
    Z01 = model.operator_seq[0].forward(X0)
    A01 = model.operator_seq[1].forward(Z01)
    Z02 = model.operator_seq[2].forward(A01)
    A02 = model.classifier_function.forward(Z02)

    X1 = X[Y[:,0]==1]  # 1类
    Z11 = model.operator_seq[0].forward(X1)
    A11 = model.operator_seq[1].forward(Z11)
    Z12 = model.operator_seq[2].forward(A11)
    A12 = model.classifier_function.forward(Z12)

    X2 = X[Y[:,0]==2]  # 2类
    Z21 = model.operator_seq[0].forward(X2)
    A21 = model.operator_seq[1].forward(Z21)
    Z22 = model.operator_seq[2].forward(A21)
    A22 = model.classifier_function.forward(Z22)


    fig = plt.figure(figsize=(10,10))

    # 第一行 -----------
    ax = fig.add_subplot(3, 3, 1)
    ax.scatter(X0[:,0], X0[:,1], marker='s', label="0", s=10)
    ax.scatter(X1[:,0], X1[:,1], marker='o', label="1", s=10)
    ax.scatter(X2[:,0], X2[:,1], marker='^', label="2", s=10)
    ax.grid()
    ax.set_aspect(1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a)原始样本")

    ax = fig.add_subplot(3, 3, 2, projection='3d')
    ax.scatter(Z01[:,0], Z01[:,1], Z01[:,2], marker='s', label="0", s=10)
    ax.scatter(Z11[:,0], Z11[:,1], Z11[:,2], marker='o', label="1", s=10)
    ax.scatter(Z21[:,0], Z21[:,1], Z21[:,2], marker='^', label="2", s=10)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_zlabel("$z_3$")
    ax.set_title("(b)空间平移(线性层1)")

    ax = fig.add_subplot(3, 3, 3, projection="3d")
    ax.scatter(A01[:,0], A01[:,1], A01[:,2], marker='s', label="0", s=10)
    ax.scatter(A11[:,0], A11[:,1], A11[:,2], marker='o', label="1", s=10)
    ax.scatter(A21[:,0], A21[:,1], A21[:,2], marker='^', label="2", s=10)
    ax.set_xlabel("$a_1$")
    ax.set_ylabel("$a_2$")
    ax.set_zlabel("$a_3$")
    ax.set_title("(c)空间扭曲(激活)")

    # 第二行 -----------
    # ax = fig.add_subplot(3, 3, 5, projection="3d")
    # ax.scatter(X0[:,0], X0[:,1], Z02[:,0], marker='s', label="0", s=10)
    # ax.scatter(X1[:,0], X1[:,1], Z12[:,1], marker='o', label="1", s=10)
    # ax.scatter(X2[:,0], X2[:,1], Z22[:,2], marker='^', label="2", s=10)
    # ax.set_xlabel("$x_1$")
    # ax.set_ylabel("$x_2$")
    # ax.set_zlabel("$z$")
    # ax.set_title("4.线性层2输出")

    ax = fig.add_subplot(3, 3, 5, projection="3d")
    ax.scatter(Z02[:,0], Z02[:,1], Z02[:,2], marker='s', label="0", s=10)
    ax.scatter(Z12[:,0], Z12[:,1], Z12[:,2], marker='o', label="1", s=10)
    ax.scatter(Z22[:,0], Z22[:,1], Z22[:,2], marker='^', label="2", s=10)
    ax.set_xlabel("$z_0$")
    ax.set_ylabel("$z_1$")
    ax.set_zlabel("$z_2$")
    ax.set_title("(d)线性层2输出")

    # 第三行 -----------
    ax = fig.add_subplot(3, 3, 7, projection="3d")
    ax.scatter(X0[:,0], X0[:,1], A02[:,0], marker='s', label="0", s=10)
    ax.scatter(X1[:,0], X1[:,1], A12[:,0], marker='o', label="1", s=10)
    ax.scatter(X2[:,0], X2[:,1], A22[:,0], marker='^', label="2", s=10)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$a_0$")
    ax.set_title("(e)分类输出类别 0")

    ax = fig.add_subplot(3, 3, 8, projection="3d")
    ax.scatter(X0[:,0], X0[:,1], A02[:,1], marker='s', label="0", s=10)
    ax.scatter(X1[:,0], X1[:,1], A12[:,1], marker='o', label="1", s=10)
    ax.scatter(X2[:,0], X2[:,1], A22[:,1], marker='^', label="2", s=10)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$a_1$")
    ax.set_title("(f)分类输出类别 1")

    ax = fig.add_subplot(3, 3, 9, projection="3d")
    ax.scatter(X0[:,0], X0[:,1], A02[:,2], marker='s', label="0", s=10)
    ax.scatter(X1[:,0], X1[:,1], A12[:,2], marker='o', label="1", s=10)
    ax.scatter(X2[:,0], X2[:,1], A22[:,2], marker='^', label="2", s=10)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$a_2$")
    ax.set_title("(g)分类输出类别 2")

    plt.show()


# 以下代码是为了展示神经网络的工作原理
def show_meaningful_weight(model, data_loader: DataLoader_11):
    count = 50

    X, Y = data_loader.get_train()  # 已经标准化好的数据

    X0 = X[Y[:,0]==0]  # 0类
    Z01 = model.operator_seq[0].forward(X0)
    A01 = model.operator_seq[1].forward(Z01)
    Z02 = model.operator_seq[2].forward(A01)
    A02 = model.classifier_function.forward(Z02)

    X1 = X[Y[:,0]==1]  # 1类
    Z11 = model.operator_seq[0].forward(X1)
    A11 = model.operator_seq[1].forward(Z11)
    Z12 = model.operator_seq[2].forward(A11)
    A12 = model.classifier_function.forward(Z12)

    X2 = X[Y[:,0]==2]  # 2类
    Z21 = model.operator_seq[0].forward(X2)
    A21 = model.operator_seq[1].forward(Z21)
    Z22 = model.operator_seq[2].forward(A21)
    A22 = model.classifier_function.forward(Z22)
    fig = plt.figure(figsize=(12,4))


    ax = fig.add_subplot(1, 3, 1)
    # 计算各个类别样本的三维均值
    classes = ['0', '1', '2']
    x = np.arange(len(classes))
    width = 0.2
    class0 = x
    class1 = x + width
    class2 = x + 2 * width
    mean0 = np.mean(Z02, axis=0)
    mean1 = np.mean(Z12, axis=0)
    mean2 = np.mean(Z22, axis=0)
    ax.bar(class0, [mean0[0],mean1[0],mean2[0]], width=width)
    ax.bar(class1, [mean0[1],mean1[1],mean2[1]], width=width)
    ax.bar(class2, [mean0[2],mean1[2],mean2[2]], width=width)
    # 计算各个类别样本的三维均值
    # classes = ['0', '1', '2']
    # mean0 = np.mean(A02, axis=0)
    # mean1 = np.mean(A12, axis=0)
    # mean2 = np.mean(A22, axis=0)
    # ax.bar(class0, [mean0[0],mean1[0],mean2[0]], width=width)
    # ax.bar(class1, [mean0[1],mean1[1],mean2[1]], width=width)
    # ax.bar(class2, [mean0[2],mean1[2],mean2[2]], width=width)
    ax.grid()
    ax.set_title("(a)线性层2分类均值")


    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(A01[:,0], A01[:,1], A01[:,2], marker='s', label="0")
    ax.scatter(A11[:,0], A11[:,1], A11[:,2], marker='o', label="1")
    ax.scatter(A21[:,0], A21[:,1], A21[:,2], marker='^', label="2")
    ax.set_xlabel("$a_1$")
    ax.set_ylabel("$a_2$")
    ax.set_zlabel("$a_3$")
    ax.set_title("(b)第二层权重$W$的含义1")

    # 生成两个分割平面
    W = model.operator_seq[2].WB.W
    B = model.operator_seq[2].WB.B
    w1 = (W[0,1] - W[0,0])/(W[2,0]-W[2,1])
    w2 = (W[1,1] - W[1,0])/(W[2,0]-W[2,1])
    b = (B[0,1] - B[0,0])/(W[2,0]-W[2,1])
    # 准备空间数据
    x = np.linspace(-1,1,count)
    y = np.linspace(-1,1,count)
    X, Y = np.meshgrid(x, y)
    Z = w1 * X + w2 * Y + b
    ax.plot_surface(X,Y,Z, alpha=0.5)

    w1 = (W[0,1] - W[0,2])/(W[2,2]-W[2,1])
    w2 = (W[1,1] - W[1,2])/(W[2,2]-W[2,1])
    b = (B[0,1] - B[0,2])/(W[2,2]-W[2,1])
    # 准备空间数据
    x = np.linspace(-1,1,count)
    y = np.linspace(-1,1,count)
    X, Y = np.meshgrid(x, y)
    Z = w1 * X + w2 * Y + b
    ax.plot_surface(X,Y,Z, alpha=0.5)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)


    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(A01[:,0], A01[:,1], A01[:,2], marker='s', label="0")
    ax.scatter(A11[:,0], A11[:,1], A11[:,2], marker='o', label="1")
    ax.scatter(A21[:,0], A21[:,1], A21[:,2], marker='^', label="2")
    ax.set_xlabel("$a_1$")
    ax.set_ylabel("$a_2$")
    ax.set_zlabel("$a_3$")
    ax.set_title("(c)第二层权重$W$的含义2")
    # 画法线
    ax.plot((0, W[0,0]), (0, W[1,0]), (0, W[2,0]), c='b')
    ax.plot((0, W[0,1]), (0, W[1,1]), (0, W[2,1]), c='r')
    ax.plot((0, W[0,2]), (0, W[1,2]), (0, W[2,2]), c='g')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    plt.show()


if __name__=="__main__":
    np.set_printoptions(suppress=True, precision=4)
    data_loader = load_data("train11-stone.txt")
    model = build_model()
    model.load("model_11_stone")
    show_result(model, data_loader)
    show_working_process(model, data_loader)
    show_meaningful_weight(model, data_loader)

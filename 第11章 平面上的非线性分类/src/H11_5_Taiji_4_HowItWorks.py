import numpy as np
from common.DataLoader_11 import DataLoader_11
import matplotlib.pyplot as plt
from H11_5_Taiji_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=10)


def show_result(model: Sequential, data_loader: DataLoader_11):
    count = 100
    # 原始样本点
    X, Y = data_loader.get_train()
    X = data_loader.de_StandardScaler_X(X)
    X0 = X[Y[:,0]==0]
    X1 = X[Y[:,0]==1]
    X2 = X[Y[:,0]==2]
    X3 = X[Y[:,0]==3]
    # 计算图片渲染元素
    x = np.linspace(-5, 5, count)
    y = np.linspace(-5, 5, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    pred_x = data_loader.StandardScaler_pred_X(input)
    output = model.predict(pred_x)
    output_max = np.max(output, axis=1, keepdims=True)
    Z = output_max.reshape(count, count)

    plt.figure(figsize=(5, 5))
    plt.contourf(X, Y, Z, cmap=plt.cm.hot)

    plt.scatter(X0[:,0], X0[:,1], marker='o', label='0')
    plt.scatter(X1[:,0], X1[:,1], marker='x', label='1')
    plt.scatter(X2[:,0], X2[:,1], marker='.', label='2')
    plt.scatter(X3[:,0], X3[:,1], marker='^', label='3')
    plt.legend(loc="right")
    plt.grid()
    plt.show()

def show_result_z2_a2(model, data_loader):
    count = 50
    # 计算图片渲染元素
    x = np.linspace(-5, 5, count)
    y = np.linspace(-5, 5, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.reshape(count*count,1),Y.reshape(count*count,1)))
    pred_x = data_loader.StandardScaler_pred_X(input)
    #A2 = model.predict(pred_x)
    #Z2 = A2 * (1-A2)

    Z1 = model.operator_seq[0].forward(pred_x)
    A1 = model.operator_seq[1].forward(Z1)
    Z2 = model.operator_seq[2].forward(A1)
    A2 = model.classifier_function.forward(Z2)

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1,5,1, projection='3d')
    Z = Z2[:,0].reshape(count,count)
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.5)
    ax.contour(X, Y, Z)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("线性输出类别0")
    ax = fig.add_subplot(1,5,2, projection='3d')
    Z = Z2[:,1].reshape(count,count)
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.5)
    ax.contour(X, Y, Z)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("线性输出类别1")
    ax = fig.add_subplot(1,5,3, projection='3d')
    Z = Z2[:,2].reshape(count,count)
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.5)
    ax.contour(X, Y, Z)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("线性输出类别2")
    ax = fig.add_subplot(1,5,4, projection='3d')
    Z = Z2[:,3].reshape(count,count)
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.5)
    ax.contour(X, Y, Z)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("线性输出类别3")

    ax = fig.add_subplot(1,5,5, projection='3d')
    Z = A2.max(axis=1, keepdims=True).reshape(count,count)
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("四分类输出")

    plt.show()

# 以下代码是为了展示神经网络的工作原理
def show_working_process(model, class_id):
    count = 100
    x = np.linspace(-5, 5, count)
    y = np.linspace(-5, 5, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    pred_x = data_loader.StandardScaler_pred_X(input)

    Z1 = model.operator_seq[0].forward(pred_x)
    A1 = model.operator_seq[1].forward(Z1)
    Z2 = model.operator_seq[2].forward(A1)

    # 隐层线性输出斜平面
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = Z1[:,idx].reshape(count, count)
            ax.plot_surface(X,Y,Z,cmap='rainbow')
            title = str.format("${0:.2f}x_1+{1:.2f}x_2+{2:.2f}$", 
                               model.operator_seq[0].WB.W[0, idx], 
                               model.operator_seq[0].WB.W[1, idx], 
                               model.operator_seq[0].WB.B[0, idx])
            ax.set_title(title)
    plt.show()
    # 激活输出
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = A1[:,idx].reshape(count, count)
            ax.plot_surface(X,Y,Z,cmap='rainbow')
            title = str.format("$w={0:.2f}$", model.operator_seq[2].WB.W[idx, class_id])
            ax.set_title(title)
    plt.show()
    # 分步叠加
    fig = plt.figure(figsize=(12,8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(3, 4, idx+1, projection='3d')
            Z = np.zeros((idx+1, count, count))
            for k in range(idx+1):
                Z[k] = A1[:,k].reshape(count, count) * model.operator_seq[2].WB.W[k, class_id]
            ax.plot_surface(X, Y, np.sum(Z, axis=0), cmap='rainbow', alpha=0.5)
            ax.contour(X, Y, np.sum(Z, axis=0))
            ax.set_title(str(idx+1))
    plt.show()


def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "data", file_name)
    data_loader = DataLoader_11(file_path)
    data_loader.load_data()
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

if __name__=="__main__":
    data_loader = load_data("train11-taiji.txt")
    model = build_model()
    model.load("model_11_taiji")
    #show_result(model, data_loader)
    #show_working_process(model, 0)
    # 最后一步线性输出和分类输出
    show_result_z2_a2(model, data_loader)

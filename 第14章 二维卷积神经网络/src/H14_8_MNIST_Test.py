import os
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Module import Sequential
import matplotlib.pyplot as plt

from H14_8_MNIST_Train import build_model, load_minist_data

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def test_model(data_loader: DataLoader_14, model: Sequential):
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))
    
    y_predict = model.forward(x)
    predict = np.argmax(y_predict, axis=1)
    label = np.argmax(test_label, axis=1)
    result = (predict == label)
    wrong_idxes = np.where(result == False)[0]

    # # 看前 10 个
    fig = plt.figure(figsize=(8, 3))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1)
        id = wrong_idxes[i]
        label_id = label[id]
        predict_id = predict[id]
        img = test_x[id].reshape(28, 28)
        ax.imshow(img, cmap="gray_r")
        ax.set_title("%d(%d)"%(label_id, predict_id))
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    # 混淆矩阵
    y_pred = np.argmax(y_predict, axis=1, keepdims=True)
    y_label = np.argmax(test_label, axis=1, keepdims=True)
    # 手工计算混淆矩阵值
    confusion_matrix = np.zeros((10, 10))
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_label[i]:  # 预测正确
            confusion_matrix[y_label[i], y_label[i]] += 1 # 对角线，True Positive
        else:  # 预测错误
            confusion_matrix[y_label[i], y_pred[i]] += 1 # FN,FP,TN
    print(confusion_matrix)
    plt.imshow(confusion_matrix, cmap='autumn_r')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, "%d"%(confusion_matrix[i, j]), ha='center', va='center')
    num_local = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.xticks(num_local)
    plt.yticks(num_local)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    plt.show()



def find_class_n(y, class_id, count):
    pos = []
    for i in range(y.shape[0]):
        if y[i] == class_id:
            pos.append(i)
            if len(pos) == count:
                return pos


def show_kernel_and_result(dataloader, model):
    kernel = model.operator_seq[0].WB.W
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(8,3))
    # 绘制卷积核，归一化数据便于比较
    names = ["-轮廓","-下侧边缘","-右侧边缘","-左斜轮廓","-右偏下","-较窄下边","-右斜轮廓","-右下移"]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    kernel_count = kernel.shape[0]
    for i in range(kernel_count):
        ax = axes[0,i]
        ax.imshow(kernel[i ,0], cmap="gray_r", interpolation="gaussian")
        ax.set_title(str(i+1)+names[i])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    

    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    y = np.argmax(test_label, axis=1)
    conv_result = model.operator_seq[0].forward(x)
    relu_result = model.operator_seq[1].forward(conv_result)
    pool_result = model.operator_seq[2].forward(relu_result)
    # 以数字 0 为例
    class_id = 0
    pos = find_class_n(y, class_id, kernel_count)
    # 原始图片
    for i in range(kernel_count):
        ax = axes[1,i]
        ax.imshow(x[pos[2], 0], cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("原始图片")

    # 卷积结果
    for i in range(kernel_count):
        ax = axes[2,i]
        ax.imshow(conv_result[pos[2], i], cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("卷积结果")

    # relu结果
    for i in range(kernel_count):
        ax = axes[3,i]
        ax.imshow(relu_result[pos[2], i], cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("激活结果")

    # pool结果
    for i in range(kernel_count):
        ax = axes[4,i]
        ax.imshow(pool_result[pos[2], i], cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("池化结果")

    plt.show()


def conv_shape(data_loader):
    kernel = np.array([
        [ 0,  1,  1, -1, -1],
        [ 1,  0, -1, -1, -1],
        [-1,  0, -1, -1,  0],
        [ 0, -1, -1,  1,  1],
        [-1,  0,  1,  1,  1],
    ])
    print("模仿第四个卷积核人为生成的卷积核:")
    print(kernel)
    conv = layer.Conv2d((1,28,28), (1,5,5))
    relu = layer.Relu()
    X, Y = data_loader.get_test()
    Y = Y.argmax(axis=1)
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(8,2))
    for i in range(10):
        # 原始样本
        poss = find_class_n(Y, i, 2)
        x = X[poss[0], 0]
        ax = axes[0, i]
        ax.imshow(x, cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("原始样本")
        # 卷积
        z = np.zeros((24,24))
        conv._conv2d(x, kernel, 24, 24, z)
        ax = axes[1, i]
        ax.imshow(z, cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("卷积结果")
        # Relu
        ax = axes[2, i]
        r = relu(z)
        ax.imshow(r, cmap="gray_r")
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        if i == 0:
            ax.set_title("激活结果")


    plt.show()

if __name__=="__main__":
    np.set_printoptions(suppress=True)
    model = build_model()
    data_loader = load_minist_data()

    model.load("MNIST_conv_14_7_8_6552") # 8
    # test_model(data_loader, model)
    # show_kernel_and_result(data_loader, model)

    conv_shape(data_loader)

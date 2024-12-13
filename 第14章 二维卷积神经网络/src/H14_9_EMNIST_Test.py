import os
import time
import math
import numpy as np
from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

from H14_9_EMNIST_Train import load_EMNIST_data, build_model
from H14_9_ShowData import find_class_n

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def load_EMNIST_data():
    print("Loading data...")
    file_path = os.path.join(os.getcwd(), "Data/ch14/EMNIST/")
    data_loader = DataLoader_14(file_path, file_path)
    data_loader.load_EMNIST_data("image")
    data_loader.StandardScaler_X(is_image=True)
    data_loader.to_onehot(26)
    return data_loader

def find_wrong_class_n(y, wrong_idx, class_id, count):
    pos = []
    for i in wrong_idx:
        if y[i] == class_id:
            pos.append(i)
            if len(pos) == count:
                return pos

def rot_img(img):
    new_img = np.flip(np.rot90(img, 3), axis=1)
    return new_img

def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    print("Running evaluation...")
    model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.testing(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))
    print("Statistic result...")
    y_predict = model.forward(x)
    predict = np.argmax(y_predict, axis=1)
    label = np.argmax(test_label, axis=1)
    result = (predict == label)
    wrong_idxes = np.where(result == False)[0]

    # # 看前 26 个判别错的测试集样本
    fig, axes = plt.subplots(nrows=2, ncols=13, figsize=(8,4))
    for i in range(26):
        poss = find_wrong_class_n(label, wrong_idxes, i, 2)
        pos = poss[0]
        ax = axes[i//13,i%13]
        label_id = label[pos]
        predict_id = predict[pos]
        img = test_x[pos].reshape(28, 28)
        img = np.flip(np.rot90(img, 3), axis=1)
        ax.imshow(img, cmap="gray_r")
        ax.set_title(chr(label_id+65) + "(" + chr(predict_id+65) + ")")
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    # 混淆矩阵
    y_pred = np.argmax(y_predict, axis=1, keepdims=True)
    y_label = np.argmax(test_label, axis=1, keepdims=True)
    # 手工计算混淆矩阵值
    confusion_matrix = np.zeros((26, 26))
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_label[i]:  # 预测正确
            confusion_matrix[y_label[i], y_label[i]] += 1 # 对角线，True Positive
        else:  # 预测错误
            confusion_matrix[y_label[i], y_pred[i]] += 1 # FN,FP,TN
    print(confusion_matrix)
    plt.imshow(confusion_matrix, cmap='autumn_r')
    for i in range(26):
        for j in range(26):
            plt.text(j, i, "%d"%(confusion_matrix[i, j]), ha='center', va='center')
    name = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    tick = range(26)
    plt.xticks(tick, name)
    plt.yticks(tick, name)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    plt.show()


    # 各层的特征图，以样本 A 为例
    poss = find_class_n(label, 0, 4)
    pos = poss[1]  # 找大写 A
    x0 = x[pos:pos+1]
    plt.imshow(rot_img(x0[0,0]), cmap="gray_r")
    plt.show()
    c1 = model.operator_seq[0].forward(x0) # (1,6,28,28)
    r1 = model.operator_seq[1].forward(c1) # (1,6,28,28)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(2,6))
    for i in range(6):
        ax = axes[i]
        ax.imshow(rot_img(r1[0,i]), cmap="gray_r")
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    p1 = model.operator_seq[2].forward(r1) # (1,6,14,14)
    c2 = model.operator_seq[3].forward(p1) # (1,16,10,10)
    r2 = model.operator_seq[4].forward(c2) # (1,16,10,10)
    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(2,6))
    for i in range(8):
        for j in range(2):
            ax = axes[i,j]
            ax.imshow(rot_img(r2[0,i*2+j]), cmap="gray_r")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    p2 = model.operator_seq[5].forward(r2) # (1,16,5,5)
    flatten = model.operator_seq[6].predict(p2) #(1,400)
    l1 = model.operator_seq[7].predict(flatten) #(1,120)
    bn = model.operator_seq[8].predict(l1) #(1,120)
    r3 = model.operator_seq[9].predict(bn) #(1,120)
    plt.imshow(r3.reshape(15,8), cmap="gray_r")
    plt.show()

    l2 = model.operator_seq[10].predict(r3) #(1,84)
    r4 = model.operator_seq[11].predict(l2) #(1,84)
    plt.imshow(r4.reshape(12,7), cmap="gray_r")
    plt.show()

    l3 = model.operator_seq[12].predict(r4) #(1,26)


    # 26 个字母的 12X7
    letters_pos = []
    for i in range(26):
        poss = find_class_n(label, i, 3)
        pos = poss[1]
        letters_pos.append(pos)

    x26 = x[letters_pos]
    c1 = model.operator_seq[0].forward(x26) # (1,6,28,28)
    r1 = model.operator_seq[1].forward(c1) # (1,6,28,28)
    p1 = model.operator_seq[2].forward(r1) # (1,6,14,14)
    c2 = model.operator_seq[3].forward(p1) # (1,16,10,10)
    r2 = model.operator_seq[4].forward(c2) # (1,16,10,10)
    p2 = model.operator_seq[5].forward(r2) # (1,16,5,5)
    flatten = model.operator_seq[6].predict(p2) #(1,400)
    l1 = model.operator_seq[7].predict(flatten) #(1,120)
    bn = model.operator_seq[8].predict(l1) #(1,120)
    r3 = model.operator_seq[9].predict(bn) #(1,120)
    l2 = model.operator_seq[10].predict(r3) #(1,84)
    r4 = model.operator_seq[11].predict(l2) #(1,84)
    fig, axes = plt.subplots(nrows=2, ncols=13, figsize=(8,3))
    for i in range(2):
        for j in range(13):
            ax = axes[i, j]
            ax.imshow(r4[i*13+j].reshape(12,7), cmap="gray_r")
            ax.set_title(chr(i*13+j+65))
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()


if __name__=="__main__":
    model_name = "EMNIST_LeNet_14_8"
    start = time.time()
    data_loader = load_EMNIST_data()
    model = build_model()
    model.load(model_name)
    #train_model(data_loader, model, model_name)
    test_model(data_loader, model, model_name)
    

import os
import numpy as np
from common.DataLoader_12 import DataLoader_12
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def load_MNIST_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("image")
    #data_loader.to_onehot(5)
    #data_loader.StandardScaler_X(is_image=True)
    #data_loader.shuffle_data()
    #data_loader.split_data(0.9)
    return data_loader

def find_class_n(y, class_id, count):
    pos = []
    for i in range(y.shape[0]):
        if y[i] == class_id:
            pos.append(i)
            if len(pos) == count:
                return pos

# 显示 4X10 的样本
def show_samples(x, y):
    names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(8,4))
    for classid in range(10):
        for i in range(4):
            poss = find_class_n(y, classid, 4)
            ax = axes[i, classid]
            if i == 0:
                ax.set_title(names[classid])
            ax.imshow(x[poss[i],0], cmap="gray_r")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()


if __name__ == '__main__':
    data_loader = load_MNIST_data()
    X, Y = data_loader.get_train()

    show_samples(X, Y)

    print("---- 训练集 ----")
    np.set_printoptions(precision=3)
    print("--- X", X.shape, "---")
    print("最大值:", np.max(X))
    print("最小值:", np.min(X))
    print("均值:", np.mean(X))
    print("标准差:", np.std(X))
    print("--- Y", Y.shape, "---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("分类值:", np.unique(Y))

    X, Y = data_loader.get_test()

    #show_samples(X, Y)
    print("---- 测试集 ----")
    np.set_printoptions(precision=3)
    print("--- X", X.shape, "---")
    print("最大值:", np.max(X))
    print("最小值:", np.min(X))
    print("均值:", np.mean(X))
    print("标准差:", np.std(X))
    print("--- Y", Y.shape, "---")
    print("最大值:", np.max(Y))
    print("最小值:", np.min(Y))
    print("分类值:", np.unique(Y))
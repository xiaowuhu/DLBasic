import os
import numpy as np
from common.DataLoader_14 import DataLoader_14
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def load_EMNIST_data():
    file_path = os.path.join(os.getcwd(), "Data/ch14/EMNIST/")
    data_loader = DataLoader_14(file_path, file_path)
    data_loader.load_EMNIST_data("image")
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

def show_samples(x, y):
    fig, axes = plt.subplots(nrows=4, ncols=13, figsize=(8,4))
    for i in range(26):
        poss = find_class_n(y, i, 2)
        for j in range(2):
            pos = poss[j]
            ax = axes[i//13*2+j,i%13]
            ax.set_title(chr(int(y[pos,0])+65))
            # 先把x向右旋转90度然后左右翻转
            img = np.flip(np.rot90(x[pos,0],3), axis=1)
            ax.imshow(img, cmap="gray_r")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


if __name__ == '__main__':
    data_loader = load_EMNIST_data()
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
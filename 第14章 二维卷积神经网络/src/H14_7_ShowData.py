import os
import numpy as np
from common.DataLoader_14 import DataLoader_14
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def load_shape_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data()
    return data_loader

def show_samples(x, y):
    num_samples = 8
    names = ["circle", "rectangle"]
    fig, axes = plt.subplots(
        nrows=data_loader.num_classes, ncols=num_samples, figsize=(8,3))
    for i in range(data_loader.num_classes):
        for j in range(num_samples):
            ax = axes[i,j]
            if j == 0:
                ax.set_title(names[i])
            ax.imshow(x[i * 1000 + j+10, 0], cmap="gray_r")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()


if __name__ == '__main__':
    data_loader = load_shape_data("train_shape_2.npz", "test_shape_2.npz")
    X, Y = data_loader.get_train()

    show_samples(X, Y)

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
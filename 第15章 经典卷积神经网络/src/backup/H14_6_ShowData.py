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
    #data_loader.to_onehot(5)
    #data_loader.StandardScaler_X(is_image=True)
    #data_loader.shuffle_data()
    #data_loader.split_data(0.9)
    return data_loader

def show_samples(x, y, names):
    num_samples = 8
    fig, axes = plt.subplots(
        nrows=data_loader.num_classes, ncols=num_samples, figsize=(8,3))
    for i in range(data_loader.num_classes):
        for j in range(num_samples):
            ax = axes[i,j]
            if j == 0:
                ax.set_title(names[y[i*1000,0]])
            ax.imshow(x[i * 1000 + j+10, 0])
            ax.axis('off')

    plt.show()

def show_samples_shape_color(x,y):
    colors = ["red", "green", "blue"]
    names = ["circle","rectangle", "triangle"]
    fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(5,5))
    for i in range(9):
        name = colors[i//3] + "-" + names[i%3]
        for j in range(y.shape[0]):
            if y[j,0] == i:
                break
        
        ax[i//3,i%3].imshow(x[j].transpose(1,2,0).astype(np.int32))
        ax[i//3,i%3].set_title(name)
        ax[i//3,i%3].axis('off')
    #endfor
    plt.show()

if __name__ == '__main__':
    #data_loader = load_shape_data("train_shape_2.npz", "test_shape_2.npz")
    data_loader = load_shape_data("train_shape_4.npz", "test_shape_4.npz")
    X, Y = data_loader.get_train()

    show_samples_shape_color(X, Y)

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
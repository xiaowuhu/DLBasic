import os
import numpy as np
from common.DataLoader_14 import DataLoader_14
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def load_FashionMNIST_data():
    file_path = os.path.join(os.getcwd(), "Data/ch14/FashionMNIST/")
    data_loader = DataLoader_14(file_path, file_path)
    data_loader.load_MNIST_data("image")
    #data_loader.to_onehot(5)
    #data_loader.StandardScaler_X(is_image=True)
    #data_loader.shuffle_data()
    #data_loader.split_data(0.9)
    return data_loader


def random_rectangle():
    s = np.random.uniform(low=0.05, high=0.15) * 28 * 28
    r = np.random.uniform(low=0.25, high=4)
    h = np.sqrt(s * r)
    w = np.sqrt(s / r)
    while True:
        x, y = np.random.uniform(0, 27, size=(2))
        if x + w < 28 and y + h < 28:
            return int(x), int(y), int(w), int(h)

def flip_and_erase(X, Y):
    new_x = np.zeros_like(X)
    new_y = np.zeros_like(Y)
    j = 0
    for i in range(X.shape[0]):
        # x, y, w, h = random_rectangle()
        # X[i,0,x:x+w,y:y+h] = np.random.uniform(low=0, high=255, size=(w, h))
        #if Y[i] not in [5,7,9]:
        new_x[j] = np.flip(X[i], axis=2)
        new_y[j] = Y[i]
        j += 1
    return new_x[:j], new_y[:j] # 返回有效数据


def noise(image, var=0.1):
    gaussian_noise = np.random.normal(0, var ** 0.5, image.shape)
    noise_image = image + gaussian_noise
    return noise_image
    #return np.clip(noise_image, 0, 1)
    
def shift(x, y):
    new_x = np.zeros_like(x)
    for i in range(x.shape[0]):
        if y[i] in [0,1,2,3,4,6]: # 左右移动
            if np.random.binomial(1, p=0.5) == 0:
                new_x[i] = np.roll(x[i], -2, axis=2)
            else:
                new_x[i] = np.roll(x[i], 2, axis=2)
        else: # 上下移动
            if np.random.binomial(1, p=0.5) == 0:
                new_x[i] = np.roll(x[i], -2, axis=1)
            else:
                new_x[i] = np.roll(x[i], 2, axis=1)
#        if np.random.binomial(1, p=0.5) == 0: # 左右翻转
        new_x[i] = np.flip(new_x[i], axis=2)
    return new_x

if __name__ == '__main__':
    print("reading data ...")
    data_loader = load_FashionMNIST_data()
    X, Y = data_loader.get_train()
    print("flipping data...")
    X_aug, Y_aug = flip_and_erase(X, Y)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    ax = axes[0]
    ax.imshow(X[0,0], cmap="gray_r")
    ax = axes[1]
    ax.imshow(X_aug[0,0], cmap="gray_r")
    plt.show()

    X_new = np.concatenate((X, X_aug))
    Y_new = np.concatenate((Y, Y_aug))
    print("save data...")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_filename = os.path.join(current_dir, "data", "FashionMNIST_Train2.npz")
    np.savez(train_filename, data=X_new, label=Y_new)

    test_X, test_Y = data_loader.get_test()
    test_filename = os.path.join(current_dir, "data", "FashionMNIST_Test2.npz")
    np.savez(test_filename, data=test_X, label=test_Y)
    print("done")

    data_loader_new = DataLoader_14(train_filename, test_filename)
    data_loader_new.load_npz_data()
    X, Y = data_loader_new.get_train()
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

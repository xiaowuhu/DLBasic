import numpy as np
import os
import matplotlib.pyplot as plt
from H8_1_ShowData import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def func(x, y):
    p1 = np.power(1-x,2) * np.exp(-np.power(x,2)-np.power(y+1,2))
    p2 = 2 * (x/5 - np.power(x,3) - np.power(y,5)) * np.exp(-np.power(x,2)-np.power(y,2))
    p3 = 0.2 * np.exp(-np.power(x+1,2)-np.power(y,2))
    z = p1 - p2 - p3
    return z

def generate_data(num_center:int, num_edge:int):
    # 正态分布样本覆盖中央地区
    t = np.random.normal(0, 1, num_center)[:, np.newaxis]
    x1 = (t - np.min(t)) / (np.max(t) - np.min(t)) * 8 - 4
    t = np.random.normal(0, 1, num_center)[:, np.newaxis]
    y1 = (t - np.min(t)) / (np.max(t) - np.min(t)) * 8 - 4
    # 均匀分布样本覆盖边缘地区
    x2 = np.random.uniform(low=-4, high=-2, size=num_edge//4)[:, np.newaxis]
    y2 = np.random.uniform(low=-4, high=4, size=num_edge//4)[:, np.newaxis]
    x3 = np.random.uniform(low=2, high=4, size=num_edge//4)[:, np.newaxis]
    y3 = np.random.uniform(low=-4, high=4, size=num_edge//4)[:, np.newaxis]
    x4 = np.random.uniform(low=-2, high=2, size=num_edge//4)[:, np.newaxis]
    y4 = np.random.uniform(low=-4, high=-2, size=num_edge//4)[:, np.newaxis]
    x5 = np.random.uniform(low=-2, high=2, size=num_edge//4)[:, np.newaxis]
    y5 = np.random.uniform(low=2, high=4, size=num_edge//4)[:, np.newaxis]
    # 合并样本
    x = np.concatenate((x1, x2, x3, x4, x5))
    y = np.concatenate((y1, y2, y3, y4, y5))
    z = func(x, y)
    data = np.hstack((x, y, z))
    return data

def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f")


def show_real_surface(data_loader: DataLoader_8):
    x = np.linspace(-4, 4, 50)[:, np.newaxis]
    y = np.linspace(-4, 4, 50)[:, np.newaxis]
    X, Y = np.meshgrid(x, y)
    z = func(X, Y)
    Z = z.reshape(50, 50)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z, cmap="rainbow")
    ax.plot_wireframe(X, Y, Z, colors="gray")

    X, Y = data_loader.get_train()
    x = X[:, 0]
    y = X[:, 1]
    z = Y
    ax.scatter(x, y, z, c='r', s=1)

    plt.grid()
    plt.show()

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data_loader = DataLoader_8(filename)
    data_loader.load_data()
    return data_loader

if __name__=="__main__":
    np.random.seed(15)
    num_center, num_edge = 2000, 500
    train_data = generate_data(num_center, num_edge)
    save_data(train_data, "train8.txt")

    dl = load_data("train8.txt")
    show_real_surface(dl)


import numpy as np
import os
import matplotlib.pyplot as plt
from H10_1_ShowData import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def func(X):
    Y = np.ones_like(X)
    idx = np.where(np.abs(X-11) < 4)
    Y[idx] = 0
    return Y

def generate_data(num_size:int, function):
    x = np.linspace(0, 20, num_size)[:, np.newaxis]
    noise = np.random.normal(0, 0.1, num_size)[:, np.newaxis]
    x = x + noise
    y = func(x)
    return np.hstack((x,y))


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.4f", header="height, label")


if __name__=="__main__":
    np.random.seed(4)
    num_size = 500
    train_data = generate_data(num_size, func)
    save_data(train_data, "train10.txt")


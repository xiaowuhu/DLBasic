import numpy as np
import os
import matplotlib.pyplot as plt
from H9_1_ShowData import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def func(x):
    return np.cos(x)*x/6 + np.cos(5 * x) / 5
    #return np.cos(x) + np.cos(5 * x) / 5

def func2(x):
    return np.cos(x)*x/6
    #return np.cos(x)

def generate_data(num_size:int, function):
    x = np.linspace(0, 6, num_size)[:, np.newaxis]
    y = function(x)
    #noise = np.random.normal(0, 0.1, num_size)[:, np.newaxis]
    #y = y + noise
    data = np.hstack((x, y))
    return data


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.2f")


if __name__=="__main__":
    np.random.seed(4)
    num_size = 50
    train_data = generate_data(num_size, func)
    save_data(train_data, "train9.txt")
    train_data = generate_data(num_size, func2)
    save_data(train_data, "val9.txt")


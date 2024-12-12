import numpy as np
import os
import matplotlib.pyplot as plt
from H7_1_ShowData import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def target_function(num_size):
    x = np.random.uniform(0, 7, num_size)[:, np.newaxis]
    y = np.cos(x)
    y1 = -0.1*x + 1
    y2 = 0.1*x - 1
    y = y * y1 * y2 + 3
    return x, y

def generate_data(num_size:int):
    x, y = target_function(num_size)
    noise = np.random.normal(0, 0.1, num_size)[:, np.newaxis]
    y = y + noise
    data = np.hstack((x, y))

    return data

def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f", header="year,price(label)")


if __name__=="__main__":
    np.random.seed(4)
    num_size = 1000
    train_data = generate_data(num_size)
    save_data(train_data, "train7.txt")

    dl = load_data("train7.txt")
    show_data(dl)


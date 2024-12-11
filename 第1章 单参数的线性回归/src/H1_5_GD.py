
# 用单次迭代方式
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from H1_1_ShowData import load_data

def single_sample_gd(train_data, eta):
    w = 0
    for i in range(train_data.shape[1]):  # 遍历所有样本
        # get x and y value for one sample
        x_i = train_data[0, i]
        y_i = train_data[1, i]
        # 式（1.4.1）
        z_i = x_i * w
        # 式（1.4.3）
        d_w = (z_i - y_i) * x_i
        # 式（1.4.4）
        w = w - eta * d_w
    return w

if __name__ == '__main__':
    file_name = "train1.txt"
    train_data = load_data(file_name)
    
    w = single_sample_gd(train_data, 0.1)
    print("w=",w)

    for eta in [0.1, 0.01, 0.001, 0.0001]:
        w = single_sample_gd(train_data, eta)
        print("eta=%f, w=%f" %(eta, w))

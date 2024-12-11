
# 用单次迭代方式
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from H1_1_ShowData import load_data

def single_sample_gd(train_x, train_y, w, eta):
    for i in range(train_x.shape[0]):  # 遍历所有样本
        # get x and y value for one sample
        x_i = train_x[i]
        y_i = train_y[i]
        # 式（1.4.1）
        z_i = x_i * w
        # 式（1.4.3）
        d_w = (z_i - y_i) * x_i
        # 式（1.4.4）
        w = w - eta * d_w
    return w

def shuffle_data(train_data):
    idx = np.random.permutation(train_data.shape[1])
    new_train_x = train_data[0][idx]
    new_train_y = train_data[1][idx]
    return new_train_x, new_train_y

if __name__ == '__main__':
    file_name = "train1.txt"
    train_data = load_data(file_name)
    print("----- 相同的步长值 0.0001, 分别运行 4 次 ----- ")
    eta = 0.0001
    for i in range(4):
        w = 0
        for j in range(1000):
            new_train_x, new_train_y = shuffle_data(train_data)
            w = single_sample_gd(new_train_x, new_train_y, w, eta)
        print("w=%f" %(w))
    
    print("----- 尝试不同的步长值 ----- ")
    for i in range(3):
        print("第%i次:" %(i+1))
        for eta in [0.0001, 0.00001, 0.000001]:
            w = 0
            for j in range(1000):
                new_train_x, new_train_y = shuffle_data(train_data)
                w = single_sample_gd(new_train_x, new_train_y, w, eta)
            print("eta=%f,w=%f" %(eta, w))

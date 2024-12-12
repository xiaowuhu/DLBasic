import numpy as np
import os
import matplotlib.pyplot as plt
from H6_1_ShowData import show_data_and_w_vector

# 字段含义
X_POS = 0
Y_POS = 1
LABEL = 2

# 右上方斜线
def func1(x):
    y = x
    return y

# 左上方斜线
def func2(x):
    y = -0.5 * x + 0.75
    return y

# 左下方斜线
def func3(x):
    y = 2 * x - 0.5
    return y

# 地理位置坐标(x,y) 已经归一化
def generate_data(num_size:int):
    data = np.zeros((num_size, 3)) # x, y, label
    # 地理位置坐标
    data[:, X_POS] = np.random.uniform(low=0, high=1, size=(num_size,))
    data[:, Y_POS] = np.random.uniform(low=0, high=1, size=(num_size,))

    for i in range(num_size):
        x = data[i, 0]
        y = data[i, 1]
        if y > func1(x) and y > func2(x): # 汉口 1
            data[i, LABEL] = 1
        elif y > func3(x) and y < func2(x): # 汉阳 2
            data[i, LABEL] = 2           
        else:
            data[i, LABEL] = 0  # 武昌 0

    return data


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f", header="x,y,label")


if __name__=="__main__":
    np.random.seed(10)
    num_size = 1000
    train_data = generate_data(num_size)
    save_data(train_data, "train6.txt")
    index = np.random.choice(np.arange(num_size), 10)
    print(train_data[index])
    show_data_and_w_vector()

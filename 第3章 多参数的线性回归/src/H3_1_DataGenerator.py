import numpy as np
import os
from H3_1_ShowData import show_data

# 生成模拟的房价数据，面积 -> 价格
def generate_data(num_size, unit_price, distance_price, bias):
    X1 = np.random.normal(scale=1, size=(num_size,))
    # 面积缩放到 50-150
    house_area = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1)) * 100 + 50
    print(np.max(house_area), np.min(house_area))

    X2 = np.random.normal(scale=1, size=(num_size,))
    # 距离缩放到 2-20
    house_distance = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2)) * 18 + 2
    print(np.max(house_distance), np.min(house_distance))

    noise = np.random.normal(scale=1, size=(num_size,)) * 25
    price = unit_price * house_area + distance_price * (20 - house_distance) + bias + noise

    data = np.zeros((num_size, 3))
    data[:, 0] = house_area
    data[:, 1] = house_distance
    data[:, 2] = price

    return data


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f", header="area,distance,price(label)")


if __name__=="__main__":
    np.random.seed(10)
    num_size = 1000
    # 单价，万
    unit_price = 5
    dist_price = 2
    bias = 10
    train_data = generate_data(num_size, unit_price, dist_price, bias)
    save_data(train_data, "train3.txt")
    show_data()
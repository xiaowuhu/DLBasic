
import numpy as np
import os

from H2_1_ShowData import show_data

# 生成模拟的房价数据，面积 -> 价格
def generate_data(num_size: int, unit_price: int, bias: int):
    a = np.random.normal(size=num_size)
    # 面积
    house_area = (a - np.min(a)) / (np.max(a) - np.min(a)) * 100 + 50
    house_area.sort()
    # 价格噪音
    b = np.random.normal(size=num_size)
    price_noise = (b - np.min(b)) / (np.max(b) - np.min(b)) * 50 - 25
    # 价格
    house_price = house_area * unit_price + bias + price_noise

    house_area_price = np.zeros((2,num_size), dtype=np.int64)
    house_area_price[0] = house_area
    house_area_price[1] = house_price

    return house_area_price


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%d", header="1:area,2:price(label)")


if __name__=="__main__":
    np.random.seed(10)
    num_size = 100
    # 单价，万
    unit_price = 2.5
    bias = 50
    train_data = generate_data(num_size, unit_price, bias)
    save_data(train_data, "train2.txt")
    index = np.random.choice(np.arange(100), 10)
    print(train_data[:,index])
    show_data()

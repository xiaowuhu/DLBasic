
import numpy as np
import matplotlib.pyplot as plt
import os

# 生成模拟的房价数据，面积 -> 价格

def generate_data(num_size, unit_price):
    # 面积
    house_area = np.random.randint(low=50, high=150, size=(num_size,))
    # 价格噪音
    price_noise = np.random.randint(low = -20, high = 50, size = (num_size,))
    # 价格
    house_price = house_area * unit_price + price_noise

    print(house_area)
    print(house_price)

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
    np.random.seed(15)
    num_size = 10
    # 单价，万
    unit_price = 2 
    train_data = generate_data(num_size, unit_price)
    #save_data(train_data, "train1.txt")

    # np.random.seed(10)
    # test_data = generate_data(num_size, unit_price)
    # save_data(test_data, "test.txt")

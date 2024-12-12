import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from H5_1_ShowData import show_data

# 房屋位置分割线
def func(x):
    y = x + 0.5
    return y

# 字段含义
X_POS = 0
Y_POS = 1
AREA = 2
SCHOOL = 3
SUN = 4
PRICE = 5

# 地理位置坐标(x,y), 在 20x20 平方公里内，以左下角为原点，右上角为(20,20)
# y = x + 10, 大于此线者为学区房，小于此线者为非学区房
# 学区房价格 = 面积 * 单价 * 1.3 + bias，非学区房价格 = 面积 * 单价 + bias
# x, y, 是否学区房, 面积, 总价格
# 可以根据 x,y 做线性分类试验，也可以根据面积、总价格做学区房分类试验
# 生成模拟的房价数据，面积 -> 价格
def generate_data(num_size:int, unit_price:int, special_price: bool, bias:int):
    data = np.zeros((num_size, 6)) # x, y, area, school, sun, price
    # 地理位置坐标
    data[:, X_POS] = np.random.uniform(low=0, high=1, size=(num_size,))
    data[:, Y_POS] = np.random.uniform(low=0, high=1, size=(num_size,))
    # 价格
    cc = 0
    for i in range(num_size):
        # 随机设置是否朝阳
        data[i, SUN] = np.random.binomial(1, 0.5)
        if data[i, Y_POS] > func(data[i, X_POS]):
            data[i, SCHOOL] = 1 # 学区房
            data[i, AREA] = np.random.randint(low=50, high=100) # 面积，学区房小一些
            data[i, PRICE] = data[i, AREA] * unit_price * 1.3 + bias # 单价贵30%
            cc += 1
        else:
            data[i, SCHOOL] = 0 # 非学区房
            data[i, AREA] = np.random.randint(low=50, high=150) # 面积，商品房大一些
            data[i, PRICE] = data[i, AREA] * unit_price + bias
    print(cc)
    return data


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f", header="X,Y,Area,Label,South,Price")


if __name__=="__main__":
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    np.random.seed(10)
    num_size = 2000
    # 单价，万
    unit_price = 5
    dist_price = 2
    bias = 10
    train_data = generate_data(num_size, unit_price, dist_price, bias)
    save_data(train_data, "train5.txt")
    index = np.random.choice(np.arange(num_size), 10)
    print(train_data[index])
    show_data()
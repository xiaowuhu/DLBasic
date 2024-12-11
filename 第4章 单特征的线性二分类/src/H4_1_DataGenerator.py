import numpy as np
import os

# 房屋位置分割线
def func(x):
    y = x + 0.5
    return y


# 安居房：7000-14000元单价，商品房：15000-30000元/平米
def generate_data(num_pos_size:int, num_neg_size:int):
    data = np.zeros((num_pos_size + num_neg_size, 2)) # unit_price, label
    # 归一化到 0.7~1.4
    price_pos = np.random.normal(scale=1, size=(num_pos_size,))
    price_pos = (price_pos - np.min(price_pos)) / (np.max(price_pos) - np.min(price_pos)) # 0~1
    price_pos = price_pos * 0.7 + 0.7
    data[0:num_pos_size, 0] = price_pos
    data[0:num_pos_size, 1] = 1  # 安居房
    # 归一化到 1.5~3.0
    price_neg = np.random.normal(scale=1, size=(num_neg_size,))
    price_neg = (price_neg - np.min(price_neg)) / (np.max(price_neg) - np.min(price_neg)) # 0~1
    price_neg = price_neg * 1.5 + 1.5
    data[num_pos_size:, 0] = price_neg
    data[num_pos_size:, 1] = 0  # 商品房

    idx = np.random.permutation(data.shape[0])
    data = data[idx]
    return data


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    np.savetxt(filename, data, fmt="%.2f", header="Price,Label")


if __name__=="__main__":
    np.random.seed(10)
    num_pos_size = 200
    num_neg_size = 300
    train_data = generate_data(num_pos_size, num_neg_size)
    save_data(train_data, "train4.txt")
    index = np.random.choice(np.arange(num_neg_size + num_pos_size), 10)
    print(train_data[index])
    #show_data()
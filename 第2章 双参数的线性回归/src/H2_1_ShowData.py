import numpy as np
import os
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data = np.loadtxt(filename)
    return data


def show_data(w = None, b = None):
    train_data = load_data("train2.txt")
    plt.scatter(train_data[0], train_data[1])
    if b is None:
        b = 0
    if w is not None:
        minv = np.min(train_data[0])
        maxv = np.max(train_data[0])
        plt.plot([minv,maxv],[minv*w+b,maxv*w+b])
    plt.grid()
    plt.xlabel("面积（平米）")
    plt.ylabel("价格（万元）")
    plt.show()
    return train_data

if __name__=="__main__":
    train_data = show_data()
    print(train_data[1]/train_data[0])


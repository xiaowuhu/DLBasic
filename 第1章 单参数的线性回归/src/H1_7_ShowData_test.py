import numpy as np
import os
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, name)
    data = np.loadtxt(filename)
    return data

def show_data(W):
    train_data = load_data("train1.txt")
    plt.scatter(train_data[0], train_data[1])
    for w in W:
        minv = np.min(train_data[0])
        maxv = np.max(train_data[0])
        plt.plot([minv,maxv],[minv*w,maxv*w])
    plt.grid()
    plt.xlabel("面积(平米)")
    plt.ylabel("价格(万元)")
    plt.title("房屋价格与面积的关系")
    plt.show()
    return train_data

if __name__=="__main__":
    W = [2.0158, 2.0150, 2.0172, 2.0178]
    train_data = show_data(W)
    # 在显示出图形后放大局部观察四条线
    


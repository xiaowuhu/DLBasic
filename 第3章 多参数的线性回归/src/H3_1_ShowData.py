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


def show_data():
    train_data = load_data("train3.txt")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2], s=5)

    plt.grid()
    plt.xlabel("面积（平米）")
    plt.ylabel("距离（公里）")
    ax.set_zlabel("价格（万元）")
    plt.show()
    return train_data

if __name__=="__main__":
    train_data = show_data()
    index = np.random.choice(np.arange(100), 10)
    print(train_data[index])


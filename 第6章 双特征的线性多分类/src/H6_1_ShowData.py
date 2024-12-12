import numpy as np
import os
import sys
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

def show_data():
    train_data = load_data("train6.txt")
    x1 = train_data[train_data[:, 2]==0]
    x2 = train_data[train_data[:, 2]==1]
    x3 = train_data[train_data[:, 2]==2]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='+', label='0-武昌')
    plt.scatter(x2[:, 0], x2[:, 1], c='g', marker='.', label='1-汉口')
    plt.scatter(x3[:, 0], x3[:, 1], c='b', marker='*', label='2-汉阳')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    return train_data

if __name__=="__main__":
    train_data = show_data()
    index = np.random.choice(np.arange(100), 10)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    print(train_data[index])


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

def show_data(n1:int=None, n2:int=None, x=None):
    train_data = load_data("train4.txt")
    x1 = train_data[train_data[:, 1]==1]
    x2 = train_data[train_data[:, 1]==0]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[0:n1, 0], [0]*n1, c='r', marker='x', label='学区房')
    plt.scatter(x2[0:n2, 0], [0]*n2, c='b', marker='o', label='普通房')
    plt.grid()
    plt.legend(loc='upper right')

    if x is not None:
        plt.scatter(x, 0, marker="*")

    plt.show()
    return train_data

if __name__=="__main__":
    train_data = show_data(10, 20)
    index = np.random.choice(np.arange(100), 10)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    print(train_data[index])


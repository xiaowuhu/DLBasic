import os
from common.DataLoader_8 import DataLoader_8
import matplotlib.pyplot as plt
import numpy as np
from H8_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def fetch_loss(name):
    file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
    training_history = np.loadtxt(file_path)
    iteration, train_loss, train_accu, val_loss, val_accu = \
        training_history[:,0],training_history[:,1],training_history[:,2],training_history[:,3],training_history[:,4]
    return iteration, train_loss, train_accu

# 平滑处理
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    smooth = np.convolve(data, window, 'full')
    return smooth[0:data.shape[0]] # 会多出 window_size-1 个元素

def show_loss(sgd_history, momentum_hoistory):
    iteration, sgd_loss, sgd_accu = sgd_history
    iteration, mom_loss, mom_accu = momentum_hoistory
    sgd_loss = moving_average(sgd_loss, 5)
    mom_loss = moving_average(mom_loss, 5)
    sgd_accu = moving_average(sgd_accu, 5)
    mom_accu = moving_average(mom_accu, 5)

    fig = plt.figure(figsize=(9, 4.5))
    # loss
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(iteration, sgd_loss, linestyle="dotted", label="SGD")
    
    ax.plot(iteration, mom_loss, label='Momentum', marker='o', markevery=0.5)
    ax.set_xlabel("迭代次数")
    ax.set_title("误差")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    # accu
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(iteration, sgd_accu, linestyle="dotted", label='SGD')
    ax.plot(iteration, mom_accu, label='Momentu,', marker='o', markevery=0.5)
    ax.set_xlabel("迭代次数")
    ax.set_title("准确率")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    plt.show()

if __name__ == '__main__':
    sgd_loss_accu = fetch_loss("history_sgd_8_7.txt")
    momentum_loss_accu = fetch_loss("history_momentum_8_7.txt")
    show_loss(sgd_loss_accu, momentum_loss_accu)

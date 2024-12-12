import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 记录训练过程
class TrainingHistory_8(object):
    def __init__(self):
        self.iteration = []
        self.train_loss = []
        self.train_accu = []
        self.val_loss = []
        self.val_accu = []
        self.best_val_loss = np.inf
        self.best_val_accu = np.inf
        self.best_iteration = 0

    def append(self, iteration, train_loss, train_accu, val_loss, val_accu):
        self.iteration.append(iteration)
        self.train_loss.append(train_loss)
        self.train_accu.append(train_accu)
        self.val_loss.append(val_loss)
        self.val_accu.append(val_accu)
        self.history = None
        # 得到最小误差值对应的权重值
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_iteration = iteration

    def get_history(self, start=0):
        if self.history is None:
            history = np.vstack(
                (self.iteration[start:], 
                 self.train_loss[start:], 
                 self.train_accu[start:], 
                 self.val_loss[start:], 
                 self.val_accu[start:]))
            self.history = history.transpose()
        return self.history

    def get_best(self):
        return self.best_iteration, self.best_val_loss, self.W, self.B

    def save_history(self, name):
        self.get_history()
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        np.savetxt(file_path, self.history)

    def load_history(self, name):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        if os.path.exists(file_path):
            self.history = np.loadtxt(file_path)
            return self.history
        else:
            print("training history file not exist!!!")

    def show_loss(self, start=0):
        self.get_history(start=start)
        iteration, train_loss, train_accu, val_loss, val_accu = \
            self.history[:,0], self.history[:,1], self.history[:,2], self.history[:,3], self.history[:,4]

        fig = plt.figure(figsize=(9, 4.5))
        # loss
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(iteration, train_loss, label='训练集')
        ax.plot(iteration, val_loss, label='验证集', marker='o', markevery=0.3)
        ax.set_xlabel("迭代次数")
        ax.set_title("误差")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()
        # accu
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(iteration, train_accu, label='训练集')
        ax.plot(iteration, val_accu, label='验证集', marker='o', markevery=0.3)
        ax.set_xlabel("迭代次数")
        ax.set_title("准确率")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()
        plt.show()        

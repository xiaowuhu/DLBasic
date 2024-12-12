import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 记录训练过程
class TrainingHistory_5(object):
    def __init__(self, W, B):
        self.iteration = []
        self.train_loss = []
        self.train_accu = []
        self.val_loss = []
        self.val_accu = []
        self.W = W
        self.B = B
        self.best_val_loss = np.inf
        self.best_val_accu = np.inf
        self.best_iteration = 0

    def append(self, iteration, train_loss, train_accu, val_loss, val_accu, W, B):
        self.iteration.append(iteration)
        self.train_loss.append(train_loss)
        self.train_accu.append(train_accu)
        self.val_loss.append(val_loss)
        self.val_accu.append(val_accu)
        # 得到最小误差值对应的权重值
        if val_loss < self.best_val_loss:
            self.W = W
            self.B = B
            self.best_val_loss = val_loss
            self.best_iteration = iteration

    def get_history(self, start=0):
        return self.iteration[start:], self.train_loss[start:], self.train_accu[start:], self.val_loss[start:], self.val_accu[start:]
    
    def get_best(self):
        return self.best_iteration, self.best_val_loss, self.W, self.B

    # 获得当前点的前 {10} 个误差记录
    def get_avg_loss(self, iter:int, count:int=10):
        assert(iter >=0)
        start = max(0, iter-count)
        end = iter
        return self.val_loss[start:end]
    
    def show_loss(self, start=0):
        iteration, train_loss, train_accu, val_loss, val_accu = self.get_history(start=start)
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
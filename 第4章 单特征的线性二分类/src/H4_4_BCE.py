import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

# 二分类交叉熵函数
def bce_loss_function(a,y):
    p1 = y * np.log(a)  # log(a + 1e-3)
    p2 = (1-y) * np.log(1-a)
    loss = -p1 - p2
    return loss
    #return np.mean(loss, axis=0, keepdims=True)

if __name__ == '__main__':
    err = 1e-2  # avoid invalid math caculation
    a = np.linspace(0+err,1-err)
    y = np.ones_like(a)
    loss1 = bce_loss_function(a,y)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(a, loss1)
    ax.set_xlabel("a")
    ax.set_title("正类的交叉熵损失函数")
    ax.grid()

    y = np.zeros_like(a)
    loss2 = bce_loss_function(a, y)
    ax = fig.add_subplot(1,2,2)
    ax.plot(a, loss2)
    ax.set_xlabel("a")
    ax.set_title("负类的交叉熵损失函数")
    ax.grid()

    plt.show()
    # 正确的分类
    A = np.array([0.7, 0.2]).transpose()
    Y = np.array([1, 0]).transpose()
    loss = bce_loss_function(A, Y)
    print("正确的分类的损失函数值:{0} -> 平均：{1}".format(loss, np.mean(loss)))
    # 错误的分类
    Y = np.array([0, 1]).transpose()
    loss = bce_loss_function(A, Y)
    print("错误的分类的损失函数值:{0} -> 平均：{1}".format(loss, np.mean(loss)))


import numpy as np


# 多分类交叉熵损失函数
def cee_loss_m(a, y):
    p = y * np.log(a)
    sum = np.sum(-p, axis=1) # 按行（在一个样本内）求和
    loss = np.mean(sum) # 按列求所有样本的平均数
    return loss


if __name__ == "__main__":
    # 6.3.4 节
    a = np.array([[0.2, 0.5, 0.3]])
    y = np.array([[0, 1, 0]])
    loss = cee_loss_m(a, y)
    print(loss)
    # 6.3.4 节
    a = np.array([[0.3, 0.6, 0.1]])
    y = np.array([[0, 1, 0]])
    loss = cee_loss_m(a, y)
    print(loss)

    a = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    loss = cee_loss_m(a, y)
    print(loss)

    a = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    y = np.array([[0, 0, 1], [0, 1, 0]])
    loss = cee_loss_m(a, y)
    print(loss)


import numpy as np
import torch

# 三分类函数
def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return a

def cee_loss_m(a, y):
    p = y * np.log(a)
    sum = np.sum(-p, axis=1) # 按行（在一个样本内）求和
    loss = np.mean(sum) # 按列求所有样本的平均数
    return loss

if __name__ == "__main__":
    z = np.array([[-1, 0, 2],[1,2,0.0]])
    y = np.array([[0,0,1],[0,1,0]])
    a = softmax(z)
    print(a)
    print(cee_loss_m(a, y))

    loss = torch.nn.CrossEntropyLoss()
    l = loss(torch.from_numpy(z), torch.from_numpy(np.argmax(y, axis=1)))
    print(l)

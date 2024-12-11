
import numpy as np

# 均方误差函数
def mse_loss(z, y):
    return np.mean(np.square(z - y)) # 可以再除以 2

# R2 准确率函数
def r2(y, mse):
    var = np.var(y)
    accuracy = 1 - mse / var
    return accuracy

def tpn(a, y):
    result = (np.round(a) == y) # a <= 0.5 -> 0, a > 0.5 -> 1
    num_correct = result.sum()  # result 中是 True(分类正确),False（分类错误）, sum() 是True的个数
    return num_correct / a.shape[0] # 分类正确的比例（正负类都算）

def lr_decay(base_lr, gamma, epoch):
    lr = base_lr * (gamma**epoch)
    return lr

# 二分类交叉熵误差函数
# cross_entropy
def bce_loss(a, y):
    p1 = y * np.log(a+1e-5)
    p2 = (1-y) * np.log(1-a+1e-5)
    return np.mean(-(p1+p2))

# 二分类函数
def logistic(z):
    a = 1.0 / (1.0 + np.exp(-z))
    return a

if __name__=="__main__":
    # test cee_loss
    A = np.array([[0.7],[0.8]])
    Y = np.array([[1.0],[0.0]])
    print(bce_loss(A, Y))

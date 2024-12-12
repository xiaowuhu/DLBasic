
import numpy as np

# 均方误差函数
def mse_loss(z, y):
    return np.mean(np.square(z - y)) # 可以再除以 2

# R2 准确率函数
def r2(y, mse):
    var = np.var(y)
    accuracy = 1 - mse / var
    return accuracy

# 二分类的准确率计算
def tpn(a, y):
    result = (np.round(a) == y) # a <= 0.5 -> 0, a > 0.5 -> 1
    num_correct = result.sum()  # result 中是 True(分类正确),False（分类错误）, sum() 是True的个数
    return num_correct / a.shape[0] # 分类正确的比例（正负类都算）

# 多分类的准确率计算
def tpn3(a, y):
    ra = np.argmax(a, axis=1)
    ry = np.argmax(y, axis=1)
    r = (ra == ry)
    correct_rate = np.mean(r)
    return correct_rate

def lr_decay(base_lr, gamma, epoch):
    lr = base_lr * (gamma**epoch)
    return lr

# 二分类交叉熵误差函数
# cross_entropy
def cee_2_loss(a, y):
    p1 = y * np.log(a)
    p2 = (1-y) * np.log(1-a)
    return np.mean(-(p1+p2))

# 多分类交叉熵损失函数
def cee_m_loss(a, y):
    p = y * np.log(a)
    sum = np.sum(-p, axis=1) # 按行（在一个样本内）求和
    loss = np.mean(sum) # 按列求所有样本的平均数
    return loss

# 二分类函数
def logistic(z):
    a = 1.0 / (1.0 + np.exp(-z))
    return a

# 三分类函数
def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return a

   
if __name__ == "__main__":
    a = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    loss = cee_m_loss(a, y)
    print(loss)

    z = np.array([[3,1,-3],[1,-3,3],[1,1,0],[1,2,-3]])
    a = softmax(z)
    print(a)

    z = np.array([[-1, 0, 2]])
    a = softmax(z)
    print(a)

    a = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    print(tpn3(a, y))

    a = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    y = np.array([[0, 1, 0], [0, 1, 0]])
    print(tpn3(a, y))
    
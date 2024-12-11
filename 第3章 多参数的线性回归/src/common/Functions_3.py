
import numpy as np

def mse_loss(y, z):
    return np.mean(np.square(z - y)) # 可以再除以 2

def r2(y, mse):
    var = np.var(y)
    accuracy = 1 - mse / var
    return accuracy

def rmse(y, z):
    t = np.square(z - y)
    return np.sqrt(np.mean(t))

def lr_decay(base_lr, gamma, epoch):
    lr = base_lr * (gamma**epoch)
    return lr

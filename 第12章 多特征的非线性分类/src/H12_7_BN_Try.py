
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

import common.Activators as activator

def func_sigmoid(Z):
    A = activator.Sigmoid().forward(Z)
    return A

def func_d_sigmoid(Z):
    a = activator.Sigmoid().forward(Z)
    da = a * (1-a)
    return da

if __name__ == '__main__':
    x = np.array([-2,-3,-4])
    w = 2
    b = 0
    z = w * x + b
    print(f"z=", z)
    a = func_sigmoid(z)
    print(f"a=", a)
    da = func_d_sigmoid(z)
    print(f"da=", da)

    mu = np.mean(x)
    sigma2 = np.mean(np.square(x - mu))
    print(f"mu={mu}, sigma2={sigma2}")

    y = (x - mu)/np.sqrt(sigma2)
    print(f"y=", y)
    z_bn = w * y + b
    print(f"z=", z_bn)
    abn = func_sigmoid(z_bn)
    print(f"a=", abn)
    da_bn = func_d_sigmoid(z_bn)
    print(f"da=", da_bn)

    print(da_bn/da)
    






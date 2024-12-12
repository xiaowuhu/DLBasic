import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def func(x):
    y = 0.4 * x * x + 0.3 * x * np.sin(15 * x) + 0.1* np.cos(50 * x)
    return y

if __name__=="__main__":
    x = np.linspace(0, 1, 100)
    y = func(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()

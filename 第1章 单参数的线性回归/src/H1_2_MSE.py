import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

# 计算均方误差
def MSE(X, w):
    z = X[:,0] * w
    mse = np.sum((z - X[:, 1]) ** 2) / X.shape[0] / 2
    return mse

if __name__=="__main__":
    X = np.array([[1.1, 0.4], [1.4, 1.5], [2.3, 1.5]])
    W = np.linspace(0, 1.4, 100)  # 100 等分
    MSEs = []  # 存储 100 个不同的 mse 值
    for w in W:
        mse = MSE(X, w)
        MSEs.append(mse)

    min_MSE = min(MSEs)
    min_idx = MSEs.index(min_MSE)
    w = W[min_idx]

    print("最小均方误差:", min_MSE)
    print("最佳 w:", w)

    plt.plot(W, MSEs)  # 横坐标 w， 纵坐标 mse 值
    plt.grid()
    plt.xlabel("$w$")
    plt.ylabel("MSE")
    plt.show()

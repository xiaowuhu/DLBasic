import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def prepare_data(XX, YY, w, b, w_width=2, b_width=2, w_len:int=50, b_len:int=50):
    m = XX.shape[0]
    X = XX.reshape(m, 1)
    Y = YY.reshape(m, 1)
    wb_len = w_len * b_len
    W = np.linspace(w - w_width, w + w_width, w_len)
    B = np.linspace(b - b_width, b + b_width, b_len)
    W, B = np.meshgrid(W, B)
    Z = np.dot(X, W.ravel().reshape(1, wb_len)) + B.ravel().reshape(1, wb_len)
    Loss = (Z - Y)**2
    Loss = Loss.sum(axis=0, keepdims=True)/m/2
    Loss = Loss.reshape(w_len, b_len)
    return W, B, Loss

if __name__=="__main__":
    # data = load_data("train2.txt")
    data = np.array([[1., 0.5], [1.4, 1.4], [2.4, 1.4]])
    X = data[:, 0]
    Y = data[:, 1]
    weight = 0.7
    bias = 0
    W, B, Loss = prepare_data(X, Y, weight, bias)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(W, B, Loss, norm=LogNorm(), cmap='rainbow')
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(W, B, Loss, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")

    plt.show()




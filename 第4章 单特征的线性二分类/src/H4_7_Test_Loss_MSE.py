import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common.Functions_4 import logistic
from H4_1_ShowData import load_data

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def prepare_data(XX, YY, w, b, w_width=2, b_width=2, w_len:int=50, b_len:int=50):
    m = XX.shape[0]
    X = XX.reshape(m, 1)
    Y = YY.reshape(m, 1)
    wb_len = w_len * b_len
    W = np.linspace(w - w_width, w + w_width, w_len)
    B = np.linspace(b - b_width, b + b_width, b_len)
    W, B = np.meshgrid(W, B)
    Z = np.dot(X, W.ravel().reshape(1, wb_len)) + B.ravel().reshape(1, wb_len)
    A = logistic(Z)
    Loss = np.square(A - Y) / 2
    #Loss = -(Y * np.log(A+1e-5) + (1-Y) * np.log(1-A+1e-5))
    Loss = Loss.sum(axis=0, keepdims=True)/m
    Loss = Loss.reshape(w_len, b_len)
    return W, B, Loss

if __name__=="__main__":
    data = load_data("train4.txt")
    X = data[:, 0]
    Y = data[:, 1]

    data = [
        # weight, bias, w_width, b_width
        [-15.5, 23.9, 10, 10],
        [-200, 308, 50, 50]
    ]

    for i in range(2):
        weight = data[i][0]
        bias = data[i][1]
        w_width = data[i][2]
        b_width = data[i][3]
    
        W, B, Loss = prepare_data(X, Y, weight, bias, w_width=w_width, b_width=b_width)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(W, B, Loss, norm=LogNorm(), cmap='rainbow')
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        ax.set_zlabel("Loss")

        ax = fig.add_subplot(1, 2, 2)
        obj = ax.contour(W, B, Loss, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)
        if i == 1:
            ax.clabel(obj, fmt='%1.3f', inline=True)
        ax.set_xlabel("w")
        ax.set_ylabel("b")

        plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common.Activators import Tanh, Relu
from common.DataLoader_7 import DataLoader_7

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_result():
    file_name = "weight-bias-1-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P1 = np.loadtxt(file_path)
    W1 = P1[0:-1]
    B1 = P1[-1:]
    file_name = "weight-bias-2-relu.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    P2 = np.loadtxt(file_path)[:, np.newaxis]
    W2 = P2[0:-1]
    B2 = P2[-1:]
    return W1, B1, W2, B2

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_7(file_path)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    return data_loader

if __name__=="__main__":
    data_loader = load_data("train7.txt")
    X, Y = data_loader.get_train()
    m = X.shape[0]

    W1, B1, W2, B2 = load_result()
    W11 = W1[0, 0]
    W12 = W1[0, 1]

    for w_width in [5, 100]:
        # 固定 B1,W2,B2 假设 W1 中的两个参数未知，并做遍历
        w_len = 50 # 分辨率
        Wn11 = np.linspace(W11 - w_width, W11 + w_width, w_len)
        Wn12 = np.linspace(W12 - w_width, W12 + w_width, w_len)
        Wn11, Wn12 = np.meshgrid(Wn11, Wn12)

        Z11 = np.dot(X, Wn11.ravel().reshape(1, w_len*w_len)) + B1[0,0]
        Z12 = np.dot(X, Wn12.ravel().reshape(1, w_len*w_len)) + B1[0,1]
        
        A11 = Relu().forward(Z11)
        A12 = Relu().forward(Z12)

        Z2 = W2[0,0] * A11 + W2[1,0] * A12 + B2[0,0]
        Loss = (Z2 - Y) ** 2
        Loss = Loss.sum(axis=0, keepdims=True)/m/2
        Loss = Loss.reshape(w_len, w_len)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(Wn11, Wn12, Loss, norm=LogNorm(), cmap='rainbow')
        ax.set_xlabel("$w_{1,1}$")
        ax.set_ylabel("$w_{1,2}$")
        ax.set_zlabel("Loss")

        ax = fig.add_subplot(1, 2, 2)
        obj = ax.contour(Wn11, Wn12, Loss, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)
        ax.clabel(obj, inline=True, fontsize=12, fmt='%1.3f')
        ax.set_xlabel("$w_{1,1}$")
        ax.set_ylabel("$w_{1,2}$")

        plt.show()


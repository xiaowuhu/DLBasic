import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common.Activators import Tanh, Relu
from common.DataLoader_10 import DataLoader_10
from H10_3_NN_train import NN
from common.Layers import Logistic
from common.LossFunctions import CrossEntropy2

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)



def load_data():
    file_name = "train10.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_10(file_path)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

if __name__=="__main__":
    data_loader = load_data()
    model = NN()
    model.load("model_10_3")
    X, Y = data_loader.get_train()


    W11 = model.linear1.WB.W[0, 0]
    W12 = model.linear1.WB.W[0, 1]
    B1 = model.linear1.WB.B

    for w_width in [2, 5]:
        # 固定 B1,W2,B2 假设 W1 中的两个参数未知，并做遍历
        w_len = 50 # 分辨率
        Wn11 = np.linspace(W11 - w_width, W11 + w_width, w_len)
        Wn12 = np.linspace(W12 - w_width, W12 + w_width, w_len)
        Wn11, Wn12 = np.meshgrid(Wn11, Wn12)
        
        Z11 = np.dot(X, Wn11.reshape(1, w_len*w_len)) + B1[0,0]
        Z12 = np.dot(X, Wn12.reshape(1, w_len*w_len)) + B1[0,1]
        
        A11 = Tanh().forward(Z11)
        A12 = Tanh().forward(Z12)

        Z2 = model.linear2.WB.W[0,0] * A11 + model.linear2.WB.W[1,0] * A12 + model.linear2.WB.B[0,0]
        A2 = Logistic().forward(Z2)

        p1 = Y * np.log(A2)
        p2 = (1-Y) * np.log(1-A2)    
        Loss = -(p1+p2)
        Loss = Loss.sum(axis=0, keepdims=True)/X.shape[0]
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


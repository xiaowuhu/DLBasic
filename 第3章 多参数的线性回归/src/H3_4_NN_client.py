import os
from common.NeuralNet_3 import NeuralNet_3
from common.DataLoader_3 import DataLoader_3
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

# 显示回归平面和归一化后的样本点
def show_result(nn, data_loader):
    X,Y = data_loader.train_x, data_loader.train_y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # draw fitting surface
    p = np.linspace(0,1)
    q = np.linspace(0,1)
    P, Q = np.meshgrid(p, q)
    R = np.hstack((P.ravel().reshape(2500,1), Q.ravel().reshape(2500,1)))
    Z = nn.predict(R, False)
    Z = Z.reshape(50,50)
    ax.plot_surface(P,Q,Z, cmap='Reds')
    ax.scatter(X[:,0], X[:,1], Y, s=10)
    ax.set_xlabel("面积")
    ax.set_ylabel("距离")
    ax.set_zlabel("价格")
    plt.show()

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train3.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_3(file_path)
    data_loader.load_data()
    data_loader.normalize_train_data()

    print("训练神经网络...")
    batch_size = 10
    epoch = 1000
    lr = 0.01
    W = np.zeros((2,1))
    B = np.zeros((1,1))
    nn = NeuralNet_3(data_loader, W, B, lr=lr, batch_size=batch_size)
    nn.train(epoch)
    print("权重值 w =", nn.W)
    print("偏置值 b =", nn.B)

    price = nn.predict(np.array([[93,15]]))
    print("预测: 距离市中心15公里的93平米的房子价格大约是:%f(万元)" % (price.flatten()[0]))
    show_result(nn, data_loader)

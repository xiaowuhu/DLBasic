import os
from common.DataLoader_8 import DataLoader_8
import matplotlib.pyplot as plt
import numpy as np
from H8_3_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def show_loss(name):
    file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
    training_history = np.loadtxt(file_path)
    iteration, train_loss, train_accu, val_loss, val_accu = \
        training_history[:,0],training_history[:,1],training_history[:,2],training_history[:,3],training_history[:,4]
    fig = plt.figure(figsize=(9, 4.5))
    # loss
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(iteration, train_loss, label='训练集')
    ax.plot(iteration, val_loss, label='验证集', marker='o', markevery=1.5)
    ax.set_xlabel("迭代次数")
    ax.set_title("误差")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    # accu
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(iteration, train_accu, label='训练集')
    ax.plot(iteration, val_accu, label='验证集', marker='o', markevery=1.5)
    ax.set_xlabel("迭代次数")
    ax.set_title("准确率")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    plt.show()

def show_result(model, data_loader: DataLoader_8):
    count = 30
    x1 = np.linspace(-4, 4, count)[:, np.newaxis]
    x2 = np.linspace(-4, 4, count)[:, np.newaxis]
    X1, X2 = np.meshgrid(x1, x2)
    X = np.zeros((count * count, 2))
    X[:,0] = X1.ravel().reshape(1, count * count)
    X[:,1] = X2.ravel().reshape(1, count * count)
    normalized_X = data_loader.StandardScaler_pred_X(X)
    normalized_Y = model.predict(normalized_X)
    Y = data_loader.de_StandardScaler_Y(normalized_Y)
    Z = Y.reshape(count, count)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X1, X2, Z, cmap="rainbow", alpha=0.3)
    ax.plot_wireframe(X1, X2, Z, colors='gray', alpha=0.3)

    normalized_X, normalized_Y = data_loader.get_train()
    X = data_loader.de_StandardScaler_X(normalized_X)
    Z = data_loader.de_StandardScaler_Y(normalized_Y)
    ax.scatter(X[:,0], X[:,1], Z, s=5, c='r')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    data_loader = load_data()
    model = NN()
    model.load("my_model")
    show_result(model, data_loader)
    show_loss("training_history_8_3.txt")

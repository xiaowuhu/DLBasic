import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def loss_w2(x, y):
    z = 0.15 * np.square(x-2) + 3 * np.square(y-2)
    return z

def l1_w2(x, y):
    z = np.abs(x) + np.abs(y)
    return z

def loss_l1(x, y, lamda):
    z = 0.15 * np.square(x-2) + 3 * np.square(y-2)
    l1 = np.abs(x) + np.abs(y)
    return z + l1 * lamda

def two_parameter(lamda):
    count = 50
    fig = plt.figure(figsize=(8,8))

    # 绘制损失函数 -------------
    x = np.linspace(-2, 4, count)
    y = np.linspace(-2, 4, count)
    X1, Y1 = np.meshgrid(x, y)
    z = loss_w2(X1, Y1)
    J = z.reshape(count, count)
    ax = fig.add_subplot(2,2,1, projection='3d')
    ax.plot_surface(X1, Y1, J, cmap="rainbow", alpha=0.3)
    ax.contour(X1, Y1, J, linewidths=0.5)
    ax.contour(X1, Y1, J, zdir='z', offset=0, cmap="coolwarm")
    # 绘制 L1 正则
    x = np.linspace(-2, 4, count)
    y = np.linspace(-2, 4, count)
    X2, Y2 = np.meshgrid(x, y)
    z = l1_w2(X2, Y2) * lamda
    L1 = z.reshape(count, count)
    ax.plot_wireframe(X2, Y2, L1, colors="red", alpha=0.3)
    ax.contour(X2, Y2, L1, linewidths=0.5)
    ax.contour(X2, Y2, L1, zdir='z', offset=0, cmap="Reds")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_zlabel("loss")
    ax.set_zlim(0, 20)

    # 绘制等高线 -------------------
    ax = fig.add_subplot(2,2,3)
    ax.contour(X1, Y1, J, cmap="coolwarm", levels=(0.5,1,2,3,4,5,7,9,12,15,20,30))
    ax.contour(X2, Y2, L1, cmap="Reds")
    ax.grid()
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")

    # 绘制新的损失函数 -------------
    ax = fig.add_subplot(2,2,2, projection='3d')
    x = np.linspace(-2, 4, count)
    y = np.linspace(-2, 4, count)
    X3, Y3 = np.meshgrid(x, y)
    Z3 = J + L1
    ax.plot_wireframe(X3, Y3, Z3, colors="blue", alpha=0.3)
    ax.contour(X3, Y3, Z3)
    ax.contour(X3, Y3, Z3, zdir='z', offset=0, cmap="coolwarm")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_zlabel("loss")
    ax.set_zlim(0, 10)

    # 绘制新的损失函数的等高线  ------------
    ax = fig.add_subplot(2,2,4)
    ax.contour(X3, Y3, Z3, cmap="coolwarm", levels=np.logspace(-2, 2, 50), norm=LogNorm())
    ax.grid()
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")

    plt.suptitle("两个参数的 $L_1$ 正则")
    plt.show()

def two_parameter_compare(lamda_list):
    count = 100
    fig = plt.figure(figsize=(12,4.5))

    # 绘制损失函数 -------------
    ax = fig.add_subplot(1, 4, 1)
    x = np.linspace(-2, 4, count)
    y = np.linspace(-2, 4, count)
    X1, Y1 = np.meshgrid(x, y)
    J = loss_w2(X1, Y1).reshape(count, count)
    ax.contour(X1, Y1, J, levels=np.logspace(-2, 2, 20), norm=LogNorm())
    ax.grid()
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_title("原始损失函数")

    L1 = l1_w2(X1, Y1).reshape(count, count)

    for i, lamda in enumerate(lamda_list):
        ax = fig.add_subplot(1, 4, i+2)
        Z = J + L1 * lamda
        ax.contour(X1, Y1, Z, levels=np.logspace(-2, 2, 50), norm=LogNorm())
        obj = ax.contour(X1, Y1, L1 * lamda, cmap="Reds", linewidths=1, linestyles="dotted")
        ax.clabel(obj, fmt='%1.3f', inline=True)
        ax.grid()
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_title("$\lambda$="+str(lamda))

    plt.show()


if __name__=="__main__":
    lamda = 2
    two_parameter(lamda)
    lamda_list = [0.1, 0.3, 0.5]
    two_parameter_compare(lamda_list)

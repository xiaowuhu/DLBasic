import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def loss_two(x, y):
    J = (x-1)**2 + 2*np.sin(y-1)**2
    return J    

def l2_two(x, y):
    z = np.square(x) + np.square(y)
    return z

def two_parameter():
    count = 100
    x = np.linspace(-1, 3, count)
    y = np.linspace(-1, 3, count)
    X1, Y1 = np.meshgrid(x, y)
    z = loss_two(X1, Y1)
    Z1 = z.reshape(count, count)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(2,2,1, projection='3d')
    ax.plot_surface(X1, Y1, Z1, cmap="rainbow", alpha=0.3)
    #ax.plot_wireframe(X1, Y1, Z1, colors="gray")
    ax.contour(X1, Y1, Z1)
    ax.contour(X1, Y1, Z1, zdir='z', offset=0, cmap="Blues")
    
    x = np.linspace(-2, 2, count)
    y = np.linspace(-2, 2, count)
    X2, Y2 = np.meshgrid(x, y)
    z = l2_two(X2, Y2)
    Z2 = z.reshape(count, count)
    ax.plot_wireframe(X2, Y2, Z2, colors="red", alpha=0.3)
    ax.contour(X2, Y2, Z2)
    ax.contour(X2, Y2, Z2, zdir='z', offset=0, cmap="Reds")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_zlabel("loss")

    ax = fig.add_subplot(2,2,3)
    ax.contour(X1, Y1, Z1, cmap="Blues")
    ax.contour(X2, Y2, Z2, cmap="Reds")
    ax.grid()
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")

    ax = fig.add_subplot(2,2,2, projection='3d')
    x = np.linspace(-2, 3, count)
    y = np.linspace(-2, 3, count)
    X3, Y3 = np.meshgrid(x, y)
    Z3 = Z1 + Z2
    ax.plot_wireframe(X3, Y3, Z3, colors="blue", alpha=0.3)
    ax.contour(X3, Y3, Z3)
    ax.contour(X3, Y3, Z3, zdir='z', offset=0, cmap="coolwarm")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_zlabel("loss")

    ax = fig.add_subplot(2,2,4)
    ax.contour(X3, Y3, Z3, cmap="coolwarm")
    ax.grid()
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")

    plt.suptitle("两个参数的 $L_2$ 正则")
    plt.show()

if __name__=="__main__":
    two_parameter()

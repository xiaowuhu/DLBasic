import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def loss_one(x):
    return np.cos(1.5 * x) + np.square(x)/3

def l2_one(x):
    return np.square(x)

def one_parameter(lamda):
    x = np.linspace(0, 2.5, 50)
    y = loss_one(x)
    plt.plot(x,y, linestyle="solid", label="$Loss(w)$")

    l2_y = lamda * l2_one(x)
    plt.plot(x, l2_y, linestyle="dotted", label="$L_2(\lambda=0.5)$")

    plt.plot(x, y + l2_y, linestyle="dashdot", label="$Loss(w)+L_2$")
    plt.grid()
    plt.legend()
    plt.ylim(0, 2)
    plt.xlabel("$w$")
    plt.ylabel("Loss")
    plt.show()

def one_parameter_compare(lamda_list):
    x = np.linspace(0, 2.5, 20)
    y = loss_one(x)
    plt.plot(x,y, linestyle="solid", label="$Loss(w)$")
    ls = ["dashed", "dotted", "dashdot"]
    for i, lamda in enumerate(lamda_list):
        l2_y = lamda * l2_one(x)
        plt.plot(x, y + l2_y, linestyle=ls[i], label="$Loss(w)+L_2(\lambda="+str(lamda)+")$")
    plt.grid()
    plt.legend()
    plt.ylim(0, 2)
    plt.xlabel("$w$")
    plt.ylabel("Loss")
    plt.show()


if __name__=="__main__":

    lamda = 0.5
    one_parameter(lamda)

    lamda_list = [0.1, 0.3, 0.5]
    one_parameter_compare(lamda_list)

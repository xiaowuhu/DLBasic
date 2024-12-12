import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


if __name__=="__main__":
    x1 = np.linspace(0, 1, 100)
    x2 = x1 * x1
    x3 = x2 * x1
    x4 = x3 * x1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(x1, linestyle='solid', label="$x$")
    ax.plot(x2, linestyle='dashed', label="$x^2$")
    ax.plot(x3, linestyle='dashdot', label="$x^3$")
    ax.plot(x4, linestyle='dotted', label="$x^4$")
    ax.legend()
    ax.grid()

    ax = fig.add_subplot(1, 4, 2)
    ax.plot(-x1 + 2 * x2)
    ax.grid()
    ax.set_title(r"$-x+2x^2$")

    ax = fig.add_subplot(1, 4, 3)
    ax.plot(x1-2*x2+x3)
    ax.set_title(r"$x-2x^2+x^3$")
    ax.grid()

    ax = fig.add_subplot(1, 4, 4)
    ax.plot(x1+x2-2*x3-2*x4)
    ax.set_title(r"$x+x^2-2x^3-2x^4$")
    ax.grid()

    plt.show()

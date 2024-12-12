
import numpy as np
import matplotlib.pyplot as plt
import common.Optimizers as Optimizers
from matplotlib.colors import LogNorm


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def function(x1, x2):
    p1 = np.square(1.5 - x1 + x1 * x2)
    p2 = np.square(2.25 - x1 + x1 * np.square(x2))
    p3 = np.square(2.625 - x1 + x1 * np.power(x2,3))
    return p1 + p2 + p3

def de_beale(x):
    x1 = x[0]
    x2 = x[1]
    dx1 = 2 * (1.5 - x1 + x1 * x2) * (x2 - 1) \
        + 2 * (2.25 - x1 + x1 * np.square(x2)) * (np.square(x2) - 1) \
        + 2 * (2.625 - x1 + x1 * np.power(x2, 3)) * (np.power(x2, 3)-1)

    dx2 = 2 * (1.5 - x1 + x1 * x2) * x1 \
        + 4 * (2.25 - x1 + x1 * np.square(x2)) * x1 * x2 \
        + 6 * (2.625 - x1 + x1 * np.power(x2, 3)) * x1 * np.square(x2)

    return np.array([dx1, dx2])

def draw_loss(W4, title):
    count = 100
    x = np.linspace(0, 3.1, count)
    y = np.linspace(-2, 1.6, count)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, levels=np.logspace(-5,5,50), norm=LogNorm(), colors='lightgray')
    ax.grid()
    ax.scatter(3, 0.5, marker='*', s=100)

    ls = ["solid", "dotted", "dashed", "dashdot", "solid"]
    for i in range(len(W4)):
        ax.plot(W4[i][0], W4[i][1], marker='.', label=title[i].replace("Momentum", "SGDM"), linestyle=ls[i])
    ax.legend()
    plt.show()


def run_different_optimizer(opt_name, start_points, lrs):
    assert(len(opt_name) == len(lrs))
    target = np.array([3, 0.5])
    W4 = []
    title = []
    for idx in range(len(lrs)):
        optimizer = Optimizers.Optimizer.create_optimizer(opt_name[idx])
        w = start_points
        lr = lrs[idx]
        W0 = []
        W1 = []
        for i in range(500):
            W0.append(w[0])
            W1.append(w[1])
            dw = de_beale(w)
            w = optimizer.update(lr, w, dw)
            if np.allclose(w, target, atol=1e-2, rtol=1e-2):
                print("stop at step %d, %s(%.2f)"%(i, opt_name[idx], lr))
                break
        W = np.vstack((np.array(W0), np.array(W1)))
        W4.append(W)
        
        title.append("%s(%.2f)"%(opt_name[idx],lr))
    
    draw_loss(W4, title)


if __name__=="__main__":
    start_point = np.array([1, 1.5])
    lrs = [0.02, 0.01, 0.005]  # 0.02 is the best
    opt_name = ["SGD", "SGD", "SGD"]
    run_different_optimizer(opt_name, start_point, lrs)

    lrs = [0.01, 0.01, 0.01]
    opt_name = [("Momentum", 0.8), ("Momentum", 0.7), ("Momentum", 0.6)] # 0.8 is the best
    run_different_optimizer(opt_name, start_point, lrs)

    lrs = [0.5, 0.7, 0.9] # 0.7 is the best
    opt_name = ['AdaGrad', "AdaGrad", "AdaGrad"]
    run_different_optimizer(opt_name, start_point, lrs)

    lrs = [0.2, 0.2, 0.2]
    opt_name = [('RMSProp',0.95), ("RMSProp", 0.8), ("RMSProp", 0.7)] # 0.9 is the best
    run_different_optimizer(opt_name, start_point, lrs)

    lrs = [0.2, 0.3, 0.4] # 0.4 is the best
    opt_name = ["Adam", "Adam", "Adam"]
    run_different_optimizer(opt_name, start_point, lrs)

    lrs = [0.02, 0.01, 0.2, 0.9, 0.4]
    opt_name = ['SGD', ("Momentum", 0.8), ('RMSProp',0.95), "AdaGrad", "Adam"]
    run_different_optimizer(opt_name, start_point, lrs)

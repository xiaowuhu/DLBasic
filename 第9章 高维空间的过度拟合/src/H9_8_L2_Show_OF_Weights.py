import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def draw_norm(ax, name, layer):
    file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name)
    WB = np.loadtxt(file_path)
    W = WB[0:-1]
    B = WB[-1:]
    print(W)
    if layer == 4:
        W = W.reshape(1,-1)
    print(W.shape[1])
    norm_W = np.linalg.norm(W, 2, axis=0)/W.shape[0]
    max_norm = np.max(norm_W)

    count = W.shape[1]
    theta = 2 * np.pi / count
    X = []
    Y = []
    for i in range(count):
        L = norm_W[i]
        degree = theta * i
        x = L * np.sin(degree)
        y = L * np.cos(degree)
        ax.plot([0,x],[0,y],color="r")
        X.append(x)
        Y.append(y)
    ax.fill(X, Y, "g", alpha=0.3)
    ax.grid()
    ax.set_xlim(-max_norm,max_norm)
    ax.set_ylim(-max_norm,max_norm)
    ax.set_aspect(1)
    ax.set_title("第%d层权重范数"%layer)

if __name__=="__main__":

    name = [
        "over_fitting_model_Linear_0.txt", 
        "over_fitting_model_Linear_1.txt", 
        "over_fitting_model_Linear_2.txt",
        "over_fitting_model_Linear_3.txt"
    ]
    fig = plt.figure(figsize=(12,3))

    for i in range(4):
        ax = fig.add_subplot(1,4,i+1)
        draw_norm(ax, name[i], i+1)
    plt.show()
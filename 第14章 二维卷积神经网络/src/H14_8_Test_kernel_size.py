import numpy as np
import common.Layers as layer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def test():
    grid = plt.GridSpec(nrows=2, ncols=3)

    x = np.zeros((7,7))
    x1 = np.array([
        [1,1,1,0,0],
        [1,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,1],
        [0,0,1,1,1]
    ]).reshape(1,1,5,5)
    x[1:6,1:6] = x1
    print("---- 原始图片 ----")
    print(x)
    x = x.reshape(1,1,7,7)
    ax = plt.subplot(grid[0:2, 0])
    ax.imshow(x[0,0], cmap="gray_r")
    ax.set_title("原始图片(7X7)")
    # 卷积核 1
    conv5 = layer.Conv2d((1, 7, 7), (1, 5, 5), stride=1, padding=0)
    w = np.array([
        [1,1,1,0,0],
        [1,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,1],
        [0,0,1,1,1]
    ]).reshape(1,1,5,5)
    conv5.WB.W = w
    print("---- 卷积核1(5X5) ----")
    print(w)
    y1 = conv5.forward(x)
    y2 = norm(y1)
    ax = plt.subplot(grid[0, 1])
    ax.imshow(y2[0,0], cmap="gray_r")
    ax.set_title("卷积核1的结果")
    ax.set_yticks([0,1,2])
    print("---- 结果1(3X3) ----")
    print(y2[0,0])

    # 卷积核2
    conv1 = layer.Conv2d((1, 7, 7), (1, 3, 3), stride=1, padding=0)
    w = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ]).reshape(1,1,3,3)
    conv1.WB.W = w
    print("---- 卷积核2,3(3X3) ----")
    print(w)
    y3 = conv1.forward(x)
    ax = plt.subplot(grid[1,1])
    ax.imshow(norm(y3)[0,0], cmap="gray_r")
    ax.set_title("卷积核2的结果")
    ax.set_yticks([0,1,2,3,4])

    # 卷积核3
    conv2 = layer.Conv2d((1, 5, 5), (1, 3, 3), stride=1, padding=0)
    conv2.WB.W = w
    y4 = conv2.forward(norm(y3))
    print("---- 结果2(5X5) ----")
    print(norm(y3))
    y5 = norm(y4)
    ax = plt.subplot(grid[1, 2])
    ax.imshow(y5[0,0], cmap="gray_r")
    ax.set_title("卷积核3的结果")
    ax.set_yticks([0,1,2])
    np.set_printoptions(precision=2)
    print("---- 结果3(3X3) ----")
    print(y5[0,0])
    plt.show()

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

if __name__=="__main__":
    test()

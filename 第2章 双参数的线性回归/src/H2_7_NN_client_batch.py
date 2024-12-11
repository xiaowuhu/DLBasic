import os
from common.NeuralNet_2_batch import NeuralNet_2_batch
from common.DataLoader_2 import DataLoader_2
import matplotlib.pyplot as plt
from H2_2_MSE import prepare_data
from H2_3_LeastSquare import calculate_w_b
from matplotlib.colors import LogNorm
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=14)

def draw_contour(ax, data_loader):
    # 绘制损失函数等高线
    X = data_loader.train_x
    Y = data_loader.train_y
    # 得到准确值作为绘图参考基准
    w_truth, b_truth = calculate_w_b(X, Y)
    W, B, Loss = prepare_data(X, Y, w_truth, b_truth, 2, 2, 100, 100)
    obj = ax.contour(W, B, Loss, levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap="Greys")
    plt.clabel(obj)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()
    data_loader.normalize_train_data()

    fig = plt.figure()

    print("训练神经网络...")
    batch_size = [1, 10, 100]
    epoch = [100, 100, 100]
    lr = [0.1, 0.2, 1]
    check_point = [100, 10, 1]

    for i in range(3):
        w, b = 0, 0
        nn = NeuralNet_2_batch(data_loader, w, b, lr[i], batch_size=batch_size[i])
        W, B = nn.train_test(epoch[i], checkpoint=check_point[i], add_start=True)
        print("权重值 w = %f, 偏置值 b = %f" % (nn.w, nn.b))
        print(len(W))

        # 绘图
        ax = fig.add_subplot(1, 3, i+1)
        ax.title.set_text("batch=%d lr=%0.1f" % (batch_size[i], lr[i]))
        draw_contour(ax, data_loader) # 绘制等高线
        # 绘制 w,b 轨迹
        #ax.plot(W, B, "ob:")
        ax.plot(W, B)
    plt.show()

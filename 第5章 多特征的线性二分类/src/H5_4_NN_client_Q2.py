import os
from common.NeuralNet_5 import NeuralNet_5
from common.DataLoader_5 import DataLoader_5
from H5_1_ShowData import show_data, load_data
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_5 import TrainingHistory_5

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train5.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_5(file_path)
    data_loader.load_data([2, 5, 3])  # 面积，总价，学区房标签
    data_loader.normalize_train_data() # 归一化
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_5):
    batch_size = 10
    epoch = 500
    lr = 0.1
    W = np.zeros((2,1))
    B = np.zeros((1,1))
    nn = NeuralNet_5(data_loader, W, B, lr=lr, batch_size=batch_size)
    training_history = nn.train(epoch, checkpoint=80)
    return nn, training_history

# 根据面积(x)和总价(y)绘制分界线
def show_result(data_loader: DataLoader_5, W, B):
    X, Y = data_loader.get_train()
    x1 = X[Y[:,0]==1]
    x2 = X[Y[:,0]==0]
    # 只显示 50 个样本
    plt.scatter(x1[0:50, 0], x1[0:50, 1], c='r', marker='+', label='学区房')
    # 只显示 510 个样本
    plt.scatter(x2[0:100, 0], x2[0:100, 1], c='b', marker='.', label='普通房')
    # 计算分界线
    w = - W[0, 0] / W[1, 0]
    b = - B[0, 0] / W[1, 0]
    x = np.linspace(0, 1, 10)[:, np.newaxis]
    y = w * x + b
    plt.plot(x, y, linestyle='dashed')
    plt.grid()
    plt.legend()
    plt.xlabel("面积")
    plt.ylabel("总价")
    plt.show()


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    data_loader = load_data()  
    print("训练神经网络...")
    nn, training_history = train(data_loader)
    iteration, val_loss, W, B = training_history.get_best()
    print("权重值 w =", W)
    print("偏置值 b =", B)
    # training_history.show_loss()
    show_result(data_loader, W, B)
    # 验证，看看预测分类结果是否与表 5.4.1 中的标签值一致
    x = np.array([
        [59, 305],
        [65, 432.5],
        [113, 575],
        [54, 361],
    ])
    pred_x = data_loader.normalize_pred_data(x)
    pred_y = nn.forward(pred_x)
    for i in range(x.shape[0]):
        if pred_y[i, 0] > 0.5:
            print("面积 %.2f 总价 %.2f 单价 %.2f 分类概率值 %f, 是学区房" %(x[i,0], x[i,1], x[i,1]/x[i,0], pred_y[i,0]))
        else:
            print("面积 %.2f 总价 %.2f 单价 %.2f 分类概率值 %f, 不是学区房" %(x[i,0], x[i,1], x[i,1]/x[i,0], pred_y[i,0]))



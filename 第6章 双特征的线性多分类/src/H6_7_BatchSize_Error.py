import os
from common.NeuralNet_6_7 import NeuralNet_6_7
from common.DataLoader_6 import DataLoader_6
import matplotlib.pyplot as plt
import numpy as np
from common.TrainingHistory_6 import TrainingHistory_6

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_6(file_path)
    data_loader.load_data([0, 1, 2]) # 加载 横坐标，纵坐标，分类标签
    data_loader.to_onehot(3) # 标签变为 onehot
    # data_loader.normalize_train_data()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def train(data_loader: DataLoader_6):
    epoch = 500
    lr = 0.5
    dw_means = []  # dw 的平均值
    db_means = []  # db 的平均值
    losses = []   # 训练结束后的误差
    test_batch_size = [1,2,4,8,10,16,20,24,28,32,36,40,48,56,64,128] # 试验的批大小
    for batch_size in test_batch_size:
        print("---- batch_size=%d----"%batch_size)
        W = np.zeros((2,3)) # 初始化为 0
        B = np.zeros((1,3))
        nn = NeuralNet_6_7(data_loader, W, B, lr=lr, batch_size=batch_size)
        training_history, dw, db = nn.train(epoch, checkpoint=80)
        iteration, val_loss, W, B = training_history.get_best()
        # 计算每一步迭代的 w,b 的模
        dw = np.array(dw)
        db = np.array(db)
        dwmean = np.mean(np.linalg.norm(dw, axis=(1,2)))
        dbmean = np.mean(np.linalg.norm(db, axis=(1,2)))
        # 保存到列表中
        dw_means.append(dwmean)
        db_means.append(dbmean)
        losses.append(val_loss)
    
    fig = plt.figure(figsize=(9,4.5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(test_batch_size, dw_means, linestyle="solid", label="$dW$", marker='.')
    ax.plot(test_batch_size, db_means, linestyle="dashed", label="$dB$", marker='x')
    ax.set_xlabel("批大小")
    ax.set_ylabel("梯度值的模")
    ax.grid()
    ax.legend()
    ax = fig.add_subplot(1,2,2)
    ax.plot(test_batch_size, losses, label="$loss$", marker='.')
    ax.set_xlabel("批大小")
    ax.set_ylabel("误差")
    ax.grid()
    ax.legend()
    plt.show()

    return nn, training_history, dw, db

def save_result(nn):
    file_name = "weights6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, nn.W)
    file_name = "bias6.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    np.savetxt(file_path, nn.B)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # 准备数据
    print("加载数据...")
    data_loader = load_data()
    print("训练神经网络...")
    nn, training_history, dw, db = train(data_loader)
    # iteration, val_loss, W, B = training_history.get_best()
    # print("权重值 w =", W)
    # print("偏置值 b =", B)
    #training_history.show_loss()
    # save_result(nn)


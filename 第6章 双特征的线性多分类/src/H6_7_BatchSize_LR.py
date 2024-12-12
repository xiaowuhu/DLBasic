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
    losses = []   # 训练结束后的误差
    loss05 = []
    #test_lr = [0.5,1,2,5,10,15,20,40.0]
    test_lr = [0.05,0.1,0.2,0.5,1,1.5,2,4.0]
    test_batch_size = [1,2,4,8,16,32,64,128] # 试验的批大小
    for i in range(8):
        batch_size = test_batch_size[i]
        lr=test_lr[i]
        print("---- batch_size=%d, lr=%f----"%(batch_size,lr))
        W = np.zeros((2,3)) # 初始化为 0
        B = np.zeros((1,3))
        nn = NeuralNet_6_7(data_loader, W, B, lr=test_lr[i], batch_size=batch_size)
        training_history, dw, db = nn.train(epoch, checkpoint=80)
        iteration, val_loss, W, B = training_history.get_best()
        losses.append(val_loss)
        # 计算每一步迭代的 w,b 的模
        W = np.zeros((2,3)) # 初始化为 0
        B = np.zeros((1,3))
        nn = NeuralNet_6_7(data_loader, W, B, lr=0.05, batch_size=batch_size)
        training_history, dw, db = nn.train(epoch, checkpoint=80)
        iteration, val_loss_05, W, B = training_history.get_best()
        loss05.append(val_loss_05)
    
    plt.plot(test_batch_size, losses, marker='o', label="学习率随批大小变化")
    for i in range(8):
        plt.text(test_batch_size[i], losses[i], str(test_lr[i]))
    plt.plot(test_batch_size, loss05, marker='x', label="固定学习率0.05")
    plt.xlabel("批大小(对数)")
    plt.ylabel("误差")
    plt.xscale("log")
    plt.grid()
    plt.legend()
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


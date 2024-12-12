import os
import sys
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
    data_loader.load_data([2, 4, 5, 3]) # 面积, 朝向, 价格, 学区房标签
    return data_loader

if __name__ == '__main__':
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    data_loader = load_data()
    X, Y = data_loader.get_train()
    neg_x = X[Y[:,0]==0]
    pos_x = X[Y[:,0]==1]
    print("学区房数量", pos_x.shape[0])
    print("学区房平均房价 %.2f" % np.mean(pos_x[:, 2]))
    print("学区房平均面积 %.2f" % np.mean(pos_x[:, 0]))
    print("学区房平均朝向 %.2f" % np.mean(pos_x[:, 1]))
    print("非学区房数量 %.2f" % neg_x.shape[0])
    print("非学区房平均房价 %.2f" %np.mean(neg_x[:, 2]))
    print("非学区房平均面积 %.2f" %np.mean(neg_x[:, 0]))
    print("非学区房平均朝向 %.2f" %np.mean(neg_x[:, 1]))
    

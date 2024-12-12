import numpy as np
import os
import math
import matplotlib.pyplot as plt
from common.DataLoader_9 import DataLoader_9
from H9_3_NN_over_fitting import NN

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(train_name, val_name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename1 = os.path.join(current_dir, "data", train_name)
    filename2 = os.path.join(current_dir, "data", val_name)
    data_loader = DataLoader_9(filename1, filename2)
    data_loader.load_data()
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    return data_loader

def func(x, y):
    return np.sqrt(1 + (y/x)**2)[0]

# 计算路的长度
def compute_length(data_loader: DataLoader_9, nn: NN):
    count = 1001
    x = np.linspace(0, 6, count)[:, np.newaxis]
    x_norm = data_loader.MinMaxScaler_pred_X(x)
    y_norm = nn.predict(x_norm)
    y = data_loader.de_MinMaxScaler_Y(y_norm)
    #show_data(x, y)
    delta_x = 6 / count
    L = 0
    for i in range(count-1):
        delta_y = y[i+1] - y[i]
        l = func(delta_x, delta_y)
        L += l
    L = L * delta_x
    return L

# 计算每个样本点到回归线的 y 方向距离
def compute_distance(data_loader: DataLoader_9, nn: NN):
    count = 1001
    X_norm, norm_label = data_loader.get_train()
    y_norm = nn.predict(X_norm)
    y_pred = data_loader.de_MinMaxScaler_Y(y_norm)
    y_label = data_loader.de_MinMaxScaler_Y(norm_label)
    distance = np.abs(y_label - y_pred)
    return distance

if __name__=="__main__":
    data_loader = load_data("train9.txt", "val9.txt")
    nn = NN()
    name = "over_fitting_model"
    nn.load(name)

    print("------ 模型 %s -------"%(name))
    nn = NN()
    nn.load(name)
    L = compute_length(data_loader, nn)
    print("道路长度为 %d 米"%(L*100))
    distance = compute_distance(data_loader, nn)
    print("最大距离= %d 米"%(np.max(distance)*100))
    print("最小距离= %d 米"%(np.min(distance)*100))
    print("平均距离= %d 米"%(np.mean(distance)*100))
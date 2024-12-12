import numpy as np
import os
from common.DataLoader_9 import DataLoader_9
from common.TrainingHistory_9 import TrainingHistory_9
from common.Module import Module
import matplotlib.pyplot as plt

from H9_3_NN_over_fitting import NN

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data():
    file_name = "train9.txt"
    file_path_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    file_name = "val9.txt"
    file_path_val = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    data_loader = DataLoader_9(file_path_train, file_path_val)
    data_loader.load_data()
    return data_loader

def show_result(data_loader: DataLoader_9, model: Module):
    X, Y = data_loader.get_train()
    plt.scatter(X, Y)

    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()

    x = np.linspace(0, 6, 100)[:, np.newaxis]
    norm_x = data_loader.MinMaxScaler_pred_X(x)
    norm_y = model.predict(norm_x)
    y = data_loader.de_MinMaxScaler_Y(norm_y)
    plt.plot(x, y)
    plt.xlabel("区域长度(百米)")
    plt.ylabel("区域宽度(百米)")
    plt.grid()
    plt.show()

# [0,1] 之间的绘图
def show_result_norm(data_loader: DataLoader_9, model: Module):
    data_loader.MinMaxScaler_X()
    data_loader.MinMaxScaler_Y()
    X, Y = data_loader.get_train()
    plt.scatter(X, Y)
    norm_x = np.linspace(0, 1, 100)[:, np.newaxis]
    norm_y = model.predict(norm_x)
    print(norm_y)
    plt.plot(norm_x, norm_y)
    plt.xlabel("区域长度(百米)")
    plt.ylabel("区域宽度(百米)")
    plt.grid()
    plt.show()

if __name__=="__main__":
    data_loader = load_data()
    # 显示 loss 曲线
    training_history = TrainingHistory_9()
    training_history.load_history("training_history_9_5_L1.txt")
    training_history.show_loss()
    # 显示拟合结果
    model = NN()
    model.load("L1_model")
    show_result(data_loader, model)

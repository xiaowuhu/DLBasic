import os
import numpy as np

from common.DataLoader_12 import DataLoader_12
import matplotlib.pyplot as plt

from H12_4_Toy_Train import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", file_name)
    data_loader = DataLoader_12(file_path)
    data_loader.load_data()
    data_loader.StandardScaler_X(is_image=True)
    return data_loader


if __name__=="__main__":
    np.set_printoptions(precision=2, suppress=True)

    model = build_model()
    data_loader = load_data("train12_4.txt")
    model.load("model_12_4_Toy")

    X, Y = data_loader.get_train()
    print("------X------")
    print(data_loader.de_StandardScaler_X(X[:10]))
    print("------Standard X------")
    print(X[:10])
    Y_pred = model.forward(X[:10], is_debug=True)
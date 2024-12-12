import os
import sys
import math
import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss

from common.DataLoader_8 import DataLoader_8
from common.TrainingHistory_8 import TrainingHistory_8
from common.Module import Module
from common.HyperParameters import HyperParameters
from common.Estimators import r2
import matplotlib.pyplot as plt
from H8_3_Train import NN, load_data


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def func(x, y, z):
    return math.sqrt((z / x)**2 + (z / y)**2 + 1)

def compute(model):
    count = 501 # 用 500 也可以，但是 501 可以得到分割整齐的数值
    x = np.linspace(-4, 4, count)
    y = np.linspace(-4, 4, count)
    X, Y = np.meshgrid(x, y)
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    normalized_X = data_loader.StandardScaler_pred_X(input)
    normalized_Y = model.predict(normalized_X)
    Y = data_loader.de_StandardScaler_Y(normalized_Y)
    Z = Y.reshape(count, count)
    delta_x = (4- (-4)) / count
    delta_y = (4- (-4)) / count
    S = 0
    for i in range(count-1):
        for j in range(count-1):
            z = np.array([Z[i,j], Z[i+1,j], Z[i,j+1], Z[i+1,j+1]])
            z_min = np.min(z)
            z_max = np.max(z)
            delta_z = z_max - z_min
            s = func(delta_x, delta_y, delta_z)
            S += s
    S = S * delta_x * delta_y
    print("房屋屋顶曲面估算面积为 %.2f 平方米"%S)

if __name__=="__main__":
    data_loader = load_data()
    model = NN()
    model.load("my_model")
    compute(model)


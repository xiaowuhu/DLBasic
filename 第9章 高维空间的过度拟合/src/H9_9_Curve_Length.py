import numpy as np
import os
import math
import matplotlib.pyplot as plt
from common.DataLoader_9 import DataLoader_9
from H9_3_NN_over_fitting import NN
from H9_3_Curve_Length import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

if __name__=="__main__":
    data_loader = load_data("train9.txt", "val9.txt")
    model_name = ["over_fitting_model", "L1_model", "L2_model", "dropout_model"]
    for name in model_name:
        print("------ 模型 %s -------"%(name))
        nn = NN()
        nn.load(name)
        L = compute_length(data_loader, nn)
        print("道路长度为 %d 米"%(L*100))
        distance = compute_distance(data_loader, nn)
        print("最大距离= %d 米"%(np.max(distance)*100))
        print("最小距离= %d 米"%(np.min(distance)*100))
        print("平均距离= %d 米"%(np.mean(distance)*100))

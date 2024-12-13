import os
import numpy as np

from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Module import Sequential
import matplotlib.pyplot as plt

from H14_7_MNIST_Train_2 import build_model, load_minist_data

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


def test_model(model: Sequential):
    test_loss, test_accu = model.testing(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))


if __name__=="__main__":
    np.set_printoptions(suppress=True)
    model = build_model()
    data_loader = load_minist_data()
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)

    checkpoints = [12470,14018,14276,14448,14792,15050,15824,16082,9546]

    name = "MNIST_conv_14_7"
    for i in range(len(checkpoints)):
        new_name = name + "_" + str(checkpoints[i])
        print(checkpoints[i])
        try:
            model.load(new_name) 
            test_model(model)
        except Exception as e:
            pass

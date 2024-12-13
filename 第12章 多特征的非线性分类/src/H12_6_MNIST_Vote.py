import os
import numpy as np

from common.DataLoader_12 import DataLoader_12
import common.Layers as layer
import common.Activators as activator
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
from common.Estimators import r2, tpn2, tpn3
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

from H12_6_MNIST_Train import build_model1, build_model3, build_model2

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def load_minist_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    return data_loader

def test_model(data_loader: DataLoader_12, model: Sequential, name):
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("模型 %s - 测试集: loss %.4f, accu %.4f" %(name, test_loss, test_accu))


def inference(model, X):
    predict = model.forward(X)
    result = np.argmax(predict, axis=1, keepdims=True)
    return result

def vote(data_loader, models):
    test_x, test_label = data_loader.get_test()
    X = data_loader.StandardScaler_pred_X(test_x)

    predicts = [] # 所有模型的结果集合
    for model in models:
        predict = inference(model, X)
        predicts.append(predict)
    predicts = np.hstack(predicts) # 变成数组 10000 x 3
    vote_result = np.zeros(predicts.shape[0]) # 10000 x 1
    for i in range(predicts.shape[0]):
        bc = np.bincount(predicts[i], minlength=10)
        if np.max(bc) == 1:
            #print(results[i])
            vote_result[i] = predicts[i, 2] # 取第三个model的结果，因为它的准确率最高
        else:
            vote_result[i] = np.argmax(bc) # 取得票多的结果
    label = np.argmax(test_label, axis=1)
    final_result = (vote_result == label)
    correct = final_result.mean()
    print("投票法的结果:", correct)


if __name__=="__main__":
    data_loader = load_minist_data()
    model1 = build_model1()
    model1.load("model_12_6_SGD")
    test_model(data_loader, model1, "model_12_6_SGD")

    model2 = build_model2()
    model2.load("model_12_6_Adam")
    test_model(data_loader, model2, "model_12_6_Adam")

    model3 = build_model3()
    model3.load("model_12_6_Momentum")
    test_model(data_loader, model3, "model_12_6_Momentum")

    vote(data_loader, (model1, model2, model3))

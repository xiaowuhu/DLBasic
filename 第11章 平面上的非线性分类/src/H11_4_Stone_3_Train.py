import os
import common.Layers as layer
import common.Activators as activator
from common.DataLoader_11 import DataLoader_11
from common.Module import Sequential
from common.HyperParameters import HyperParameters
import matplotlib.pyplot as plt
from H11_1_base import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "data", file_name)
    data_loader = DataLoader_11(file_path)
    data_loader.load_data()
    data_loader.to_onehot(3)
    data_loader.StandardScaler_X()
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

def build_model():
    model = Sequential(
        layer.Linear(2, 3, init_method="xavier", optimizer="RMSProp"),
        activator.Tanh(), 
        layer.Linear(3, 3, init_method="xavier", optimizer="RMSProp"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train11-stone.txt")
    params = HyperParameters(max_epoch=5000, batch_size=32, learning_rate=0.01)
    training_history = train_model(data_loader, model, params, checkpoint=50)
    training_history.show_loss()
    #model.save("model_11_stone")

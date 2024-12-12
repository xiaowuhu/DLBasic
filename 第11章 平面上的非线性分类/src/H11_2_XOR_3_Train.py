import common.Layers as layer
import common.Activators as activator

from common.Module import Sequential
from common.HyperParameters import HyperParameters
import matplotlib.pyplot as plt

from H11_1_base import *

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def build_model():
    model = Sequential(
        layer.Linear(2, 2, init_method="kaiming", optimizer="Momentum"),
        activator.Relu(),     # 激活层
        layer.Linear(2, 1, init_method="kaiming", optimizer="Momentum"),   # 线性层2（输出层）
    )
    model.set_classifier_loss_function(layer.LogisticCrossEntropy()) # 二分类函数+交叉熵损失函数
    return model

if __name__=="__main__":
    model = build_model()
    data_loader = load_data("train11-xor.txt")
    params = HyperParameters(max_epoch=500, batch_size=32, learning_rate=0.1)
    training_history = train_model(data_loader, model, params, checkpoint=1)
    training_history.show_loss()
    #model.save("model_11_xor_Relu")

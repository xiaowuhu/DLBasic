import os
import time
import numpy as np
import matplotlib.pyplot as plt

from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Estimators import tpn3
from common.Module import Sequential, SubProcessInfo
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
from H14_6_Shape_CNN import build_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_shape_data(train_file_name, test_file_name):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data()
    data_loader.to_onehot(5)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.9)
    return data_loader

# 计算损失函数和准确率
def check_loss(
        data_loader:DataLoader_14, 
        batch_size: int, batch_id: int, 
        model: Sequential, 
        training_history:TrainingHistory, 
        epoch:int, iteration:int, 
        learning_rate:float
):
    # 训练集
    x, label = data_loader.get_batch(batch_size, batch_id)
    train_loss, train_accu = model.compute_loss_accuracy(x, label)
    # 验证集
    x, label = data_loader.get_val()
    val_loss, val_accu = model.compute_loss_accuracy(x, label)
    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    print("轮数 %d, 迭代 %d, 训练集: loss %.4f, accu %.4f, 验证集: loss %.4f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))


def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    print(model_name)
    model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))

    loss = 0
    accu = 0
    step = 100
    for i in range(0, data_loader.num_test, step):
        print(i)
        #predict = model.predict(x[i:i+step])
        predict = model.forward(x[i:i+step])
        test_loss = model.loss_function(predict, test_label[i:i+step])
        test_accu = tpn3(predict, test_label[i:i+step])
        loss += test_loss * step
        accu += test_accu * step
        print(loss, accu)
    end = time.time()
    print("测试集: loss %.4f, accu %.4f" %(loss/data_loader.num_test, accu/data_loader.num_test))

   
if __name__=="__main__":
    model:Sequential = build_model()
    data_loader = load_shape_data("train_shape.npz", "test_shape.npz")
    test_model(data_loader, model, "Shape_conv_14_6")

import os
import time
import math
import numpy as np
from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Module import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_shape_data(train_file_name, test_file_name, mode="image"):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data(mode=mode)
    data_loader.to_onehot(data_loader.num_classes)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.9)
    return data_loader

# 计算损失函数和准确率
def check_loss(
    data_loader:DataLoader_14, 
    batch_size: int, 
    batch_id: int, 
    model: Sequential, 
    training_history:TrainingHistory, 
    epoch:int, 
    iteration:int, 
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

def train(
    data_loader: DataLoader_14, 
    model: Sequential,
    params: HyperParameters,
    lrs: LRScheduler,
    model_name: str,
    checkpoint = 1,
):
    training_history = TrainingHistory()
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    check_iteration = int(batch_per_epoch * checkpoint)    
    iteration = 0 # 每一批样本算作一次迭代
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(params.batch_size, batch_id)
            batch_predict = model.forward(batch_X)
            model.backward(batch_predict, batch_label)
            model.update_parameters_value(params.learning_rate)
            iteration += 1
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
        # if epoch % 5 == 0:
        #     model.save(model_name)
    return training_history


def build_cnn_model():
    model = Sequential()
    # 两个卷积核
    c1 = layer.Conv2d((3,28,28), (2,3,3), stride=1, padding=0, optimizer="SGD")
    model.add_op(c1)
    model.add_op(layer.Relu())
    c2 = layer.Conv2d(c1.output_shape, (4,3,3), stride=1, padding=0, optimizer="SGD")
    model.add_op(c2)
    model.add_op(layer.Relu())
    p2 = layer.Pool2d(c2.output_shape, (2,2), stride=2, padding=0, pool_type="max")
    model.add_op(p2)
    linear_shape = np.prod(p2.output_shape)
    print(linear_shape)
    model.add_op(layer.Flatten(p2.output_shape, (1, linear_shape)))
    model.add_op(layer.Linear(linear_shape, 9, init_method="kaiming", optimizer="SGD"))
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    #model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    total = data_loader.num_test
    correct = math.ceil(total * test_accu)
    print("测试集: loss %.4f, accu (%i/%d) = %.2f%%" %(test_loss, correct, total, test_accu*100))

def train_model(data_loader, model, model_name, show_loss=True):
    start = time.time()
    params = HyperParameters(max_epoch=5, batch_size=64)
    lrs = LRScheduler.step_lrs(0.01, 0.9, 700)
    training_history = train(data_loader, model, params, lrs, model_name, checkpoint=0.5)
    end = time.time()
    print("用时:", end-start)
    model.save(model_name)
    training_history.show_loss()
    return 

if __name__=="__main__":
    data_loader = load_shape_data("train_shape_4.npz", "test_shape_4.npz", mode="image")
    model_name = "Shape_4_conv_14_6_CNN_1"
    model = build_cnn_model()
    train_model(data_loader, model, model_name)
    test_model(data_loader, model, model_name)
    

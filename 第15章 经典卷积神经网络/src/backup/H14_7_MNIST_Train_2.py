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

def load_minist_data():
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_loader = DataLoader_14(file_path, file_path)
    data_loader.load_MNIST_data("image")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.92)
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
    x, label = data_loader.get_val(1000)
    val_loss, val_accu = model.compute_loss_accuracy(x, label)
    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    print("轮数 %d, 迭代 %d, 训练集: loss %.4f, accu %.4f, 验证集: loss %.4f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    
    if val_accu > 0.99:
        # save model
        model_name = str.format("MNIST_conv_14_7_{0}", iteration)
        model.save(model_name)
        print("save model in: ", model_name)

def train(
        data_loader: DataLoader_14, 
        model: Sequential,
        params: HyperParameters,
        lrs: LRScheduler,
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
                need_stop = check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
                if need_stop:
                    return training_history
    return training_history

def build_model():
    model = Sequential()

    c1 = layer.Conv2d((1,28,28), (8,3,3), stride=1, padding=0, optimizer="Adam")
    model.add_op(c1)
    model.add_op(layer.Relu())
    c2 = layer.Conv2d(c1.output_shape, (16,3,3), stride=1, padding=0, optimizer="Adam")
    model.add_op(c2)
    model.add_op(layer.Relu())
    p2 = layer.Pool2d(c2.output_shape, (2,2), stride=2, padding=0, pool_type="max")
    model.add_op(p2)

    c3 = layer.Conv2d(p2.output_shape, (32,3,3), stride=1, padding=0, optimizer="Adam")
    model.add_op(c3)
    model.add_op(layer.Relu())
    c4 = layer.Conv2d(c3.output_shape, (64,3,3), stride=1, padding=0, optimizer="Adam")
    model.add_op(c4)
    model.add_op(layer.Relu())
    p4 = layer.Pool2d(c4.output_shape, (2,2), stride=2, padding=0, pool_type="max")
    model.add_op(p4)


    linear_shape = np.prod(p4.output_shape)
    model.add_op(layer.Flatten(p4.output_shape, (1, linear_shape)))
    model.add_op(layer.Linear(linear_shape, 256, init_method="kaiming", optimizer="Adam"))
    model.add_op(layer.BatchNorm1d(256))
    model.add_op(layer.Relu())
    model.add_op(layer.Linear(256, 32, init_method="kaiming", optimizer="Adam"))
    model.add_op(layer.BatchNorm1d(32))
    model.add_op(layer.Relu())
    model.add_op(layer.Dropout(dropout_ratio=0.25))
    model.add_op(layer.Linear(32, 10, init_method="kaiming", optimizer="Adam"))

    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy())
    return model

def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))

def train_model(data_loader, model, model_name, show_loss=True):
    params = HyperParameters(max_epoch=20, batch_size=64, learning_rate=0.01)
    lrs = LRScheduler.step_lrs(0.01, 0.9, 500)
    training_history = train(data_loader, model, params, lrs, checkpoint=0.1)
    end = time.time()
    print("用时:", end-start)
    model.save(model_name)
    training_history.show_loss()
    return 

if __name__=="__main__":
    model_name = "MNIST_conv_14_7_6"
    start = time.time()
    data_loader = load_minist_data()
    model = build_model()
    train_model(data_loader, model, model_name)
    test_model(data_loader, model, model_name)
    

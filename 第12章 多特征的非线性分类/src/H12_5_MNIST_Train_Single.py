import os
import math
import time
import numpy as np
from multiprocessing import shared_memory, Process, Event

from common.DataLoader_12 import DataLoader_12
import common.Layers_5 as layer
import common.Activators_5 as activator
from common.Module_5 import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
from common.Estimators import r2, tpn2, tpn3
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_minist_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.8)
    return data_loader

# 计算损失函数和准确率
def check_loss(
        data_loader:DataLoader_12, 
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
    return train_loss, train_accu, val_loss, val_accu

def train_model(
    train_data_shape,
    event_data, # 获取训练数据通知
    event_grad, # 回传梯度通知
    event_update # 获得梯度更新通知
):
    model = build_model(is_create = False)
    # 接收训练数据共享
    train_data_shm = shared_memory.SharedMemory(create=False, name="train_data")
    train_data_buf = np.ndarray(train_data_shape, dtype=np.float64, buffer=train_data_shm.buf)
    # 接收初始化权重参数
    event_update.wait()
    event_update.clear()
    model.set_parameters_value()

    iteration = 0
    while True:
        event_data.wait() # 从主控请求训练数据
        event_data.clear() # 得到数据，清空标志
        batch_X = train_data_buf[:,0:-10]
        batch_label = train_data_buf[:,-10:]
        batch_predict = model.forward(batch_X)
        model.backward(batch_predict, batch_label)
        model.share_grad_value()
        event_grad.set() # 通知主控可以拿梯度了
        event_update.wait() # 等待梯度更新数据
        event_update.clear() # 得到数据，清空标志
        model.set_parameters_value()
        iteration += 1
    return

def build_model(is_create = True):
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="SGD"),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="SGD"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    model.create_share_memory(is_create) # 在主控端建立共享参数及梯度
    return model

def main(params:HyperParameters, checkpoint=1):
    model = build_model(is_create=True)
    data_loader = load_minist_data()
    lrs = LRScheduler.step_lrs(0.1, 0.9, 5000)
    #lrs = LRScheduler.fixed_lrs(0.1)
    batch_per_epoch = math.ceil(data_loader.num_train / params.batch_size)
    # 获得共享内存大小
    X, Y = data_loader.get_batch(params.batch_size, 0)
    XY = np.hstack((X, Y))
    train_data_shm = shared_memory.SharedMemory(create=True, name="train_data", size=XY.nbytes)
    train_data_buf = np.ndarray(XY.shape, dtype=np.float64, buffer=train_data_shm.buf)
    # 通知
    event_data = Event()
    event_grad = Event()
    event_update = Event()
    event_data.clear()
    event_grad.clear()
    event_update.clear()
    p = Process(target=train_model, args=(XY.shape, event_data, event_grad, event_update,))
    p.start()

    # 共享初始化参数
    model.share_parameters_value()
    event_update.set()
    # 开始训练
    training_history = TrainingHistory()
    check_iteration = int(batch_per_epoch * checkpoint)    
    iteration = 0
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(params.batch_size, batch_id)
            train_data_buf[:,0:-10] = batch_X
            train_data_buf[:,-10:] = batch_label
            event_data.set() # 通知子进程可以拿数据了
            event_grad.wait() # 等待梯度数据
            event_grad.clear() # 得到梯度数据，清空标志
            model.get_grad_value()
            # 更新模型参数
            model.update_parameters_value(params.learning_rate)
            # 共享模型参数
            model.share_parameters_value()
            event_update.set()
            iteration += 1
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)
    training_history.show_loss()
    p.terminate()
    model.save("model_12_5")
    model.close_share_memory()
    train_data_shm.close()
    train_data_shm.unlink()


if __name__=="__main__":
    params = HyperParameters(max_epoch=10, batch_size=32)
    main(params, checkpoint=0.1)

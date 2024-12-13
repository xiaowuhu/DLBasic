import os
import math
import time
import numpy as np
import multiprocessing
from multiprocessing import shared_memory, Process

from common.DataLoader_12 import DataLoader_12
import common.Layers_5 as layer
import common.Activators_5 as activator
from common.Module_5_2 import Sequential
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
from common.Estimators import r2, tpn2, tpn3
import common.LearningRateScheduler as LRScheduler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


class SubProcessInfo(object):
    def __init__(self, id, train_data_shm, train_data_buf, process, event_data, event_grad, event_update):
        self.id = id
        self.train_data_shm = train_data_shm
        self.train_data_buf = train_data_buf
        self.process = process
        self.event_data = event_data
        self.event_grad = event_grad
        self.event_update = event_update

def load_minist_data():
    file_path = os.path.join(os.getcwd(), "Data/ch12/MNIST/")
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.9)
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
    process_id,
    train_data_shape,
    event_data, # 获取训练数据通知
    event_grad, # 回传梯度通知
    event_update # 获得参数更新通知
):
    model = build_model()
    model.setup_share_memory(process_id)
    # 接收训练数据共享
    train_data_shm = shared_memory.SharedMemory(create=False, name=str(process_id) + "_train_data")
    train_data_buf = np.ndarray(train_data_shape, dtype=np.float64, buffer=train_data_shm.buf)
    # 接收初始化权重参数
    event_update.wait()
    event_update.clear()
    model.set_parameters_value(process_id)

    iteration = 0
    while True:
        event_data.wait() # 从主控请求训练数据
        event_data.clear() # 得到数据，清空标志
        batch_X = train_data_buf[:,0:-10]
        batch_label = train_data_buf[:,-10:]
        print("---- get data:" + str(process_id), batch_X.sum())
        batch_predict = model.forward(batch_X)
        model.backward(batch_predict, batch_label)
        model.share_grad_value(process_id) 
        event_grad.set() # 通知主控可以拿梯度了
        event_update.wait() # 等待参数更新数据
        event_update.clear() # 得到数据，清空标志
        model.set_parameters_value(process_id)
        iteration += 1
    return

def build_model() -> Sequential:
    model = Sequential(
        layer.Linear(784, 64, init_method="kaiming", optimizer="Adam"),
        activator.Relu(),
        layer.Linear(64, 32, init_method="kaiming", optimizer="Adam"),
        activator.Relu(),
        layer.Linear(32, 10, init_method="kaiming", optimizer="Adam"),
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

def main(params:HyperParameters, num_process: int, lrs:LRScheduler, checkpoint=1):
    model:Sequential = build_model()
    model.create_share_memory(num_process) # 在主控端建立共享内存 for 参数及梯度
    data_loader = load_minist_data()
    big_batch_size = params.batch_size * num_process
    # 每次取 num_process * batch_size 个数据, 分别送给 num_process 个进程
    batch_per_epoch = data_loader.num_train // big_batch_size  # 丢弃最后一些样本
    # 获得训练数据共享内存大小
    X, Y = data_loader.get_batch(params.batch_size, 0)
    XY = np.hstack((X, Y))

    # 共享初始化参数
    model.share_parameters_value()
    sub_process_info = []
    for process_id in range(num_process):
        # 在主控端建立共享内存 for 训练数据
        train_data_shm = shared_memory.SharedMemory(create=True, name=str(process_id) + "_train_data", size=XY.nbytes)
        train_data_buf = np.ndarray(XY.shape, dtype=np.float64, buffer=train_data_shm.buf)
        event_data = multiprocessing.Event()
        event_grad = multiprocessing.Event()
        event_update = multiprocessing.Event()
        event_data.clear()
        event_grad.clear()
        event_update.clear()
        p = Process(target=train_model, args=(process_id, XY.shape, event_data, event_grad, event_update,))
        sub_process_info.append(SubProcessInfo(process_id, train_data_shm, train_data_buf, p, event_data, event_grad, event_update))
        p.start()
        event_update.set()
    
    time1 = time.time()

    # 开始训练
    training_history = TrainingHistory()
    check_iteration = int(batch_per_epoch * checkpoint)    
    iteration = 0
    for epoch in range(params.max_epoch):
        data_loader.shuffle_data()
        for batch_id in range(batch_per_epoch):
            batch_X, batch_label = data_loader.get_batch(big_batch_size, batch_id)
            # 共享训练数据给子进程
            for process_id in range(num_process):
                start = process_id * params.batch_size
                end = start + params.batch_size
                sub_process_info[process_id].train_data_buf[:,0:-10] = batch_X[start:end]
                sub_process_info[process_id].train_data_buf[:,-10:] = batch_label[start:end]
                sub_process_info[process_id].event_data.set() # 通知子进程可以拿训练数据了
                print("main: send data:" + str(process_id), batch_X[start:end].sum())
        
            # 等待所有子进程的梯度数据
            for process_id in range(num_process):
                sub_process_info[process_id].event_grad.wait() # 等待梯度数据
                sub_process_info[process_id].event_grad.clear() # 得到梯度数据，清空标志
            # 获得梯度数据
            model.get_grad_value(num_process)
            # 更新模型参数
            model.update_parameters_value(params.learning_rate)
            # 共享模型参数
            model.share_parameters_value()

            for process_id in range(num_process):
                sub_process_info[process_id].event_update.set() # 通知子进程可以拿参数了
            
            iteration += 1
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)

    time2 = time.time()
    print(time2 - time1)
    training_history.show_loss()
    for process_id in range(num_process):
        sub_process_info[process_id].train_data_shm.close()
        sub_process_info[process_id].train_data_shm.unlink()
        sub_process_info[process_id].process.terminate()
    #model.save("model_12_5_2")
    model.close_share_memory()


if __name__=="__main__":
    params = HyperParameters(max_epoch=10, batch_size=32)
    num_process = 8
    lrs = LRScheduler.step_lrs(0.01, 0.9, 3000)
    main(params, num_process, lrs, checkpoint=0.1)

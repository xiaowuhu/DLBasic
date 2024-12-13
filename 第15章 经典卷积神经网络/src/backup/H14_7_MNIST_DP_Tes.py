import os
import time
import math
import numpy as np
from multiprocessing import shared_memory, Event, Process
import matplotlib.pyplot as plt

from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
from common.Module import Sequential, SubProcessInfo, SubProcessInfo_for_prediction
from common.HyperParameters import HyperParameters
from common.TrainingHistory import TrainingHistory
import common.LearningRateScheduler as LRScheduler
from common.Estimators import tpn3


plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_minist_data():
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_loader = DataLoader_14(file_path, file_path)
    data_loader.load_MNIST_data("image")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)#, mean=0.5, std=0.5)
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
    x, label = data_loader.get_val(1000)
    val_loss, val_accu = model.compute_loss_accuracy(x, label)
    # 记录历史
    training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
    print("轮数 %d, 迭代 %d, 训练集: loss %.4f, accu %.4f, 验证集: loss %.4f, accu %.4f, 学习率:%.4f" \
          %(epoch, iteration, train_loss, train_accu, val_loss, val_accu, learning_rate))
    
    if val_loss < 0.04 and val_accu > 0.99:
        # save model
        model_name = str.format("MNIST_conv_14_7_{0}", iteration)
        model.save(model_name)
        print("save model in: ", model_name)
    if val_accu >= 0.995:
        return True
    else:
        return False

# 子进程运行本函数
def pred_model(
    process_id,
    test_x_shape,
    test_y_shape,
    batch_pred,
    event_data, # 获取预测数据通知
    event_pred, # 回传推理结果
    event_update # 获得推理参数广播通知
):
    model: Sequential = build_model()
    model.setup_share_memory_for_prediction(process_id, batch_pred)
    # 接收训练数据共享
    batch_size = test_x_shape[0]
    assert(test_x_shape[0] == test_y_shape[0])  # batch size
    test_data_shm = shared_memory.SharedMemory(create=False, name=str(process_id) + "_test_data")
    test_data_shape = (batch_size, np.prod(test_x_shape[1:]) + test_y_shape[1])
    test_data_buf = np.ndarray(test_data_shape, dtype=np.float64, buffer=test_data_shm.buf)
    # 接收初始化权重参数
    event_update.wait()
    event_update.clear()
    model.get_parameters_value()

    iteration = 0
    while True:
        event_data.wait() # 从主控请求训练数据
        event_data.clear() # 得到数据，清空标志
        batch_data = test_data_buf[:]
        #print("---- get data:" + str(process_id), batch_data.sum())
        batch_X = batch_data[:, 0:-test_y_shape[1]].reshape(test_x_shape)
        batch_label = batch_data[:, -test_y_shape[1]:]
        batch_predict = model.forward(batch_X)
        model.share_pred_value(process_id, batch_predict) 
        event_pred.set() # 通知主控可以拿梯度了
        iteration += 1
    return

def main(model: Sequential, 
         data_loader: DataLoader_14, 
         num_process: int, 
):
    # 计算每批的分类结果所占用的字节数
    batch_pred = np.zeros((params.batch_size, 10))  # 10 分类
    model.create_share_memory_for_prediction(num_process, batch_pred) # 在主控端建立共享内存 for 参数及预测结果
    big_batch_size = params.batch_size * num_process
    # 每次取 num_process * batch_size 个数据, 分别送给 num_process 个进程
    batch_per_epoch = math.ceil(data_loader.num_test / big_batch_size)  # 保留最后一些样本
    # 获得训练数据共享内存大小
    X, Y = data_loader.get_batch(params.batch_size, 0)
    XY = np.hstack((X.reshape(params.batch_size, -1), Y))  # for image data

    # 共享推理参数
    model.share_parameters_value()
    sub_processes_info = []
    for process_id in range(num_process):
        # 在主控端建立共享内存 for 测试数据
        test_data_shm = shared_memory.SharedMemory(create=True, name=str(process_id) + "_test_data", size=XY.nbytes)
        test_data_buf = np.ndarray(XY.shape, dtype=np.float64, buffer=test_data_shm.buf)
        event_data = Event()
        event_data.clear()
        event_pred = Event()
        event_pred.clear()
        event_update= Event()
        event_update.clear()
        p = Process(target=pred_model, 
                    args=(process_id, X.shape, Y.shape, batch_pred,
                          event_data, event_pred, event_update,))
        sub_processes_info.append(
            SubProcessInfo_for_prediction(
                process_id, p,
                test_data_shm, test_data_buf, 
                event_data, event_pred, event_update
            )
        )
        p.start()
        event_update.set()
    
    time1 = time.time()
    # 开始预测
    test_results = []
    # 准备数据buffer
    batch_data = np.zeros_like(XY)
    for batch_id in range(batch_per_epoch):
        print(batch_id)
        batch_x, batch_label = data_loader.get_batch_test(big_batch_size, batch_id)
        batch_x = data_loader.StandardScaler_pred_X(batch_x)
        # 共享训练数据给子进程
        for process_id in range(num_process):
            start = process_id * params.batch_size
            end = start + params.batch_size
            x = batch_x[start:end]
            if x.shape[0] > 0:
                x = x.reshape(x.shape[0], -1)
                # x.shape[0] 有可能比 params.batch_size 小，因为是在测试集尾端，不够64
                batch_data[0:x.shape[0],0:x.shape[1]] = x
                batch_data[0:x.shape[0],x.shape[1]:] = batch_label[start:end]
            sub_processes_info[process_id].share_test_data(batch_data)
            #print("main: share data:", batch_data.sum())
        # 等待所有子进程的梯度数据
        for process_id in range(num_process):
            sub_processes_info[process_id].wait_pred_and_clear()
        # 获得预测结果
        results = model.get_pred_value(num_process)
        batch_result = np.concatenate(results)
        test_results.append(batch_result)
    test_result_array = np.concatenate(test_results)
    test_results_10k = test_result_array[0:data_loader.num_test]
    _, test_label = data_loader.get_test()
    loss = model.loss_function(test_results_10k, test_label)    
    accu = tpn3(test_results_10k, test_label)
    print("loss - ", loss)
    print("accu - ", accu)
    time2 = time.time()
    print("用时:", time2 - time1)
    for process_id in range(num_process):
        sub_processes_info[process_id].close()
    model.close_share_memory(num_process)

def build_model():
    model = Sequential()
    # 卷积层1-Relu
    c1 = layer.Conv2d((1,28,28), (32,3,3), stride=1, padding=0, optimizer="Adam")
    model.add_op(c1)
    model.add_op(layer.Relu())
    # 卷积层2-Relu-MaxPool
    c2 = layer.Conv2d(c1.output_shape, (32,3,3), stride=1, padding=0, optimizer="SGD")
    model.add_op(c2)
    model.add_op(layer.Relu())
    p2 = layer.Pool2d(c2.output_shape, (2,2), stride=2, padding=0, pool_type="max")
    model.add_op(p2)
    # flatten
    linear_shape = np.prod(p2.output_shape)
    model.add_op(layer.Flatten(p2.output_shape, (1, linear_shape)))
    # 全连接层1-BN-Relu-全连接层2
    model.add_op(layer.Linear(linear_shape, 256, init_method="kaiming", optimizer="Adam"))
    model.add_op(layer.BatchNorm1d(256))
    model.add_op(layer.Relu())
    model.add_op(layer.Dropout(0.3))
    model.add_op(layer.Linear(256, 128, init_method="kaiming", optimizer="Adam"))
    model.add_op(layer.BatchNorm1d(128))
    model.add_op(layer.Relu())
    model.add_op(layer.Dropout(0.1))
    model.add_op(layer.Linear(128, 10, init_method="kaiming", optimizer="Adam"))

    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
    return model

def test_model(data_loader: DataLoader_14, model: Sequential):
    start = time.time()
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    loss = 0
    accu = 0
    for i in range(0, data_loader.num_test, 100):
        print(i)
        predict = model.predict(x[i:i+100])
        test_loss = model.loss_function(predict, test_label[i:i+100])
        test_accu = tpn3(predict, test_label[i:i+100])
        loss += test_loss
        accu += test_accu
    end = time.time()
    print("用时", end - start)
    print("测试集: loss %.4f, accu %.4f" %(loss/100, accu/100))

    
if __name__=="__main__":
    params = HyperParameters(max_epoch=1, batch_size=64)
    num_process = 8
    model:Sequential = build_model()
    model.load("MNIST_conv_14_7")
    data_loader = load_minist_data()
    test_model(data_loader, model)
    #main(model, data_loader, num_process)


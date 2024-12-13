import os
import time
import numpy as np
from multiprocessing import shared_memory, Event, Process
import matplotlib.pyplot as plt

from common.DataLoader_14 import DataLoader_14
import common.Layers as layer
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
    

# 子进程运行本函数
def train_model(
    process_id,
    train_x_shape,
    train_y_shape,
    event_data, # 获取训练数据通知
    event_grad, # 回传梯度通知
    event_update # 获得参数更新通知
):
    model: Sequential = build_model()
    model.setup_share_memory(process_id)
    # 接收训练数据共享
    batch_size = train_x_shape[0]
    assert(train_x_shape[0] == train_y_shape[0])  # batch size
    train_data_shm = shared_memory.SharedMemory(create=False, name=str(process_id) + "_train_data")
    train_data_shape = (batch_size, np.prod(train_x_shape[1:]) + train_y_shape[1])
    train_data_buf = np.ndarray(train_data_shape, dtype=np.float64, buffer=train_data_shm.buf)
    # 接收初始化权重参数
    event_update.wait()
    event_update.clear()
    model.get_parameters_value()

    iteration = 0
    while True:
        event_data.wait() # 从主控请求训练数据
        event_data.clear() # 得到数据，清空标志
        batch_data = train_data_buf[:]
        #print("---- get data:" + str(process_id), batch_data.sum())
        batch_X = batch_data[:, 0:-train_y_shape[1]].reshape(train_x_shape)
        batch_label = batch_data[:, -train_y_shape[1]:]
        
        batch_predict = model.forward(batch_X)
        model.backward(batch_predict, batch_label)
        model.share_grad_value(process_id) 
        event_grad.set() # 通知主控可以拿梯度了
        event_update.wait() # 等待参数更新数据
        event_update.clear() # 得到数据，清空标志
        model.get_parameters_value()
        iteration += 1
    return

def main(model: Sequential, 
         data_loader: DataLoader_14, 
         params:HyperParameters, 
         num_process: int, 
         lrs:LRScheduler, 
         checkpoint: float=1
):
    model.create_share_memory_for_training(num_process) # 在主控端建立共享内存 for 参数及梯度
    big_batch_size = params.batch_size * num_process
    # 每次取 num_process * batch_size 个数据, 分别送给 num_process 个进程
    batch_per_epoch = data_loader.num_train // big_batch_size  # 丢弃最后一些样本
    # 获得训练数据共享内存大小
    X, Y = data_loader.get_batch(params.batch_size, 0)
    XY = np.hstack((X.reshape(params.batch_size, -1), Y))  # for image data

    # 共享初始化参数
    model.share_parameters_value()
    sub_processes_info = []
    for process_id in range(num_process):
        # 在主控端建立共享内存 for 训练数据
        train_data_shm = shared_memory.SharedMemory(create=True, name=str(process_id) + "_train_data", size=XY.nbytes)
        train_data_buf = np.ndarray(XY.shape, dtype=np.float64, buffer=train_data_shm.buf)
        event_data = Event()
        event_grad = Event()
        event_update = Event()
        event_data.clear()
        event_grad.clear()
        event_update.clear()
        p = Process(target=train_model, args=(process_id, X.shape, Y.shape, event_data, event_grad, event_update,))
        sub_processes_info.append(SubProcessInfo(process_id, p, train_data_shm, train_data_buf, event_data, event_grad, event_update))
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
                batch_data = np.hstack((batch_X[start:end].reshape(params.batch_size, -1), batch_label[start:end]))
                sub_processes_info[process_id].share_train_data(batch_data)
                #print("main: share data:", batch_data.sum())
            # 等待所有子进程的梯度数据
            for process_id in range(num_process):
                sub_processes_info[process_id].wait_grad_and_clear()
            # 获得梯度数据
            model.get_grad_value(num_process)
            # 更新模型参数
            model.update_parameters_value(params.learning_rate)
            # 共享模型参数
            model.share_parameters_value()

            for process_id in range(num_process):
                sub_processes_info[process_id].share_parameter_done() # 通知子进程可以拿参数了
            
            iteration += 1
            params.learning_rate = lrs.get_learning_rate(iteration)
            if iteration==1 or iteration % check_iteration == 0:
                check_loss(data_loader,  params.batch_size, batch_id, model, training_history, epoch, iteration, params.learning_rate)

    time2 = time.time()
    print("用时:", time2 - time1)
    training_history.show_loss()
    for process_id in range(num_process):
        sub_processes_info[process_id].close()
    model.save("Shape_conv_14_6")
    model.close_share_memory(num_process)

# def build_model():
#     model = Sequential()
#     # 卷积层1-Relu
    # c1 = layer.Conv2d((1,28,28), (16,3,3), stride=1, padding=0, optimizer="SGD")
    # model.add_op(c1)
    # model.add_op(layer.Relu())

    # c2 = layer.Conv2d(c1.output_shape, (16,3,3), stride=1, padding=0, optimizer="SGD")
    # model.add_op(c2)
    # model.add_op(layer.Relu())
    # p2 = layer.Pool2d(c2.output_shape, (2,2), stride=2, padding=0, pool_type="max")
    # model.add_op(p2)

    # linear_shape = np.prod(p2.output_shape)
    # #print(linear_shape)
    # model.add_op(layer.Flatten(p2.output_shape, (1, linear_shape)))

    # model.add_op(layer.Linear(linear_shape, 512, init_method="kaiming", optimizer="Adam"))
    # model.add_op(layer.BatchNorm1d(512))
    # model.add_op(layer.Relu())
    # model.add_op(layer.Dropout(0.5))
    # model.add_op(layer.Linear(512, 64, init_method="kaiming", optimizer="Adam"))
    # model.add_op(layer.BatchNorm1d(64))
    # model.add_op(layer.Relu())
    # model.add_op(layer.Dropout(0.2))
    # model.add_op(layer.Linear(64, 5, init_method="kaiming", optimizer="Adam"))

#     model.set_classifier_loss_function(layer.SoftmaxCrossEntropy()) # 多分类函数+交叉熵损失函数
#     return model

def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    print(model_name)
    model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))

    
if __name__=="__main__":
    params = HyperParameters(max_epoch=100, batch_size=64)
    num_process = 8
    lrs = LRScheduler.step_lrs(0.01, 0.9, 20)
    model:Sequential = build_model()  # 使用 H14_6_Shape_CNN.py 中的模型
    data_loader = load_shape_data("train_shape.npz", "test_shape.npz")
    main(model, data_loader, params, num_process, lrs, checkpoint=0.5)
    test_model(data_loader, model, "Shape_conv_14_6")

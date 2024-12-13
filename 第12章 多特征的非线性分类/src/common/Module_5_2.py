from multiprocessing import shared_memory, Process, Event
import numpy as np

from .OperatorBase_5 import Operator
from . import Layers_5 as layer
from . import LossFunctions as loss
from .Estimators import r2, tpn2, tpn3

# 模型基类
class Module(object):
    def predict(self, X):
        return self.forward(X)

    # 目前这个operators必须有,否则从locals()中获取不到
    def save_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id)
            op.save(op_name)
            unique_id += 1

    def load_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id)
            op.load(op_name)
            unique_id += 1


# 顺序计算类
class Sequential(Module):
    def __init__(self, *operators):
        self.operator_seq:list[Operator] = operators
        for op in operators:
            print(op)
        self.reversed_ops = self.operator_seq[::-1]
        self.loss_function = None
        self.classifier_loss_function = None
        self.classifier_function = None
        self.net_type = "Regression"
        self.paramters_dict = self.get_parameters()

    # 分类任务设置此项为 ce2 or ce3
    def set_classifier_function(self, classifier_func):
        self.classifier_function = classifier_func

    # 设置损失函数（不应该放在初始化函数中）
    def set_loss_function(self, loss_func):
        self.loss_function = loss_func
    
    # 设置快捷反向传播函数，直接做 a-y, z-y 等等
    # 回归任务设置此项为 mse_loss
    def set_classifier_loss_function(self, combined_func):
        self.classifier_loss_function = combined_func
        if isinstance(combined_func, layer.LogisticCrossEntropy):
            self.set_classifier_function(layer.Logisitic())   # 二分类交叉熵损失函数
            self.set_loss_function(loss.CrossEntropy2())   # 二分类交叉熵损失函数
            self.net_type = "BinaryClassifier"
        if isinstance(combined_func, layer.SoftmaxCrossEntropy):
            self.set_classifier_function(layer.Softmax())   # 多分类交叉熵损失函数
            self.set_loss_function(loss.CrossEntropy3())   # 多分类交叉熵损失函数
            self.net_type = "Classifier"

    def forward(self, X, is_debug=False):
        data = X
        for op in self.operator_seq:
            data = op.forward(data)
            if is_debug:
                print(op)
                print(data)
        if self.classifier_function is not None:
            data = self.classifier_function.forward(data)
            if is_debug:
                print(self.classifier_function)
                print(data)
        return data

    def backward(self, predict, label):
        if self.classifier_loss_function is not None:
            delta = self.classifier_loss_function.backward(predict, label)
        else:
            assert(self.loss_function is not None)
            delta = self.loss_function.backward(predict, label)
        for op in self.reversed_ops:
            delta = op.backward(delta)

    def compute_loss(self, predict, label):
        assert(self.loss_function is not None)
        return self.loss_function(predict, label)

    def save(self, name):
        super().save_parameters(name, self.operator_seq)        

    def load(self, name):
        super().load_parameters(name, self.operator_seq)        

    def compute_loss_accuracy(self, x, label):
        predict = self.forward(x)
        loss = self.compute_loss(predict, label)
        if self.net_type == "Regression":
            accu = r2(label, loss)
        elif self.net_type == "BinaryClassifier":
            accu = tpn2(predict, label)
        elif self.net_type == "Classifier":
            accu = tpn3(predict, label)
        return loss, accu

    def get_parameters(self):
        param_dict = {}
        unique_id = 0
        for op in self.operator_seq:
            op_name = op.__class__.__name__ + "_" + str(unique_id)
            wb = op.get_parameters()
            if wb is not None:
                param_dict[op_name] = wb
            unique_id += 1
        return param_dict

    def update(self, lr):
        for name, wb in self.paramters_dict.items():
            wb.Update(lr)
    
    # 父进程，只调用一次，用于共享参数和接收梯度
    def create_share_memory(self, num_process):
        self.shm_buf = {}  # 存储 name -> (shm,buf)
        for process_id in range(num_process):
            self.shm_buf[process_id] = {}  # 建立二级字典
        for op_name, wb in self.paramters_dict.items():
            W, B = wb.get_WB()
            WB = np.concatenate((W, B))
            # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
            wb_shm = shared_memory.SharedMemory(name=op_name, create=True, size=WB.nbytes)
            wb_buf = np.ndarray(WB.shape, dtype=np.float64, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)
            # 在父进程端建立，用于子进程共享 WB 梯度给父进程，每个子进程都需要有一份copy
            for process_id in range(num_process):
                grad_shm_name = str(process_id) + "_" + op_name + "_grad"
                grad_shm = shared_memory.SharedMemory(name=grad_shm_name, create=True, size=WB.nbytes)
                buf = np.ndarray(WB.shape, dtype=np.float64, buffer=grad_shm.buf)
                self.shm_buf[process_id][op_name] = (grad_shm, buf)

    # 父进程：获得梯度数据共享
    def get_grad_value(self, num_process):
        for op_name, wb in self.paramters_dict.items():
            # 清空梯度
            wb.dW = wb.dW * 0
            wb.dB = wb.dB * 0
            for process_id in range(num_process):
                # 累加从不同子进程传回来的梯度
                wb.dW += self.shm_buf[process_id][op_name][1][0:-1, :]
                wb.dB += self.shm_buf[process_id][op_name][1][-1:, :]
                #print("main: get grad:" + str(process_id), self.shm_buf[shm_name][1][0:-1, :].sum())
            # 取平均梯度
            wb.dW /= num_process
            wb.dB /= num_process
            
    # 父进程：更新权重参数
    def update_parameters_value(self, lr):
        for _, wb in self.paramters_dict.items():
            wb.Update(lr)

    # 父进程：共享当前参数
    def share_parameters_value(self):
        for op_name, wb in self.paramters_dict.items():
            W, B = wb.get_WB()
            self.shm_buf[op_name][1][0:-1] = W[:]
            self.shm_buf[op_name][1][-1:] = B[:]
            #print("main: share W:" + op_name, W.sum())

    # 子进程，只调用一次，建立共享内存副本，用于接收参数和共享梯度
    def setup_share_memory(self, process_id):
        self.shm_buf = {}
        self.shm_buf[process_id] = {}
        for op_name, wb in self.paramters_dict.items():
            W, B = wb.get_WB()
            WB = np.concatenate((W, B))
            # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
            wb_shm = shared_memory.SharedMemory(name=op_name, create=False, size=WB.nbytes)
            wb_buf = np.ndarray(WB.shape, dtype=np.float64, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)
            # 在父进程端建立的用于子进程共享 WB 梯度给父进程，每个子进程都需要有一份copy
            grad_shm_name = str(process_id) + "_" + op_name + "_grad"
            grad_shm = shared_memory.SharedMemory(name=grad_shm_name, create=False, size=WB.nbytes)
            grad_buf = np.ndarray(WB.shape, dtype=np.float64, buffer=grad_shm.buf)
            self.shm_buf[process_id][op_name] = (grad_shm, grad_buf)

    # 子进程：把梯度copy到share memory中
    def share_grad_value(self, process_id):
        for op_name, wb in self.paramters_dict.items():
            dW, dB = wb.get_dWB() # W 和 B 合在一起了
            self.shm_buf[process_id][op_name][1][0:-1] = dW[:]
            self.shm_buf[process_id][op_name][1][-1:] = dB[:]
            #print("----share grad:" + str(process_id), dW.sum())

    
    # 子进程：获得权重参数
    def set_parameters_value(self, process_id):
        for op_name, wb in self.paramters_dict.items():
            wb.W = self.shm_buf[op_name][1][0:-1, :]
            wb.B = self.shm_buf[op_name][1][-1:, :]
            #print("----get W:" + str(process_id), wb.W.sum())

    def close_share_memory(self):
        for k,v in self.shm_buf.items():
            if isinstance(v, dict):
                for k1,v1 in v.items():
                    v1[0].close()
                    v1[0].unlink()
            else:
                v[0].close()
                v[0].unlink()

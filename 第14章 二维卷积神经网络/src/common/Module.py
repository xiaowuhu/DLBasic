from multiprocessing import shared_memory
import numpy as np

from .Layers import Classifier, Operator as op, LossFunctions as loss
from .Estimators import r2, tpn2, tpn3

# 模型基类
class Module(object):
    # def predict(self, X):
    #     return self.forward(X)

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
        self.operator_seq:list[op.Operator] = list(operators)
        self.reversed_ops = self.operator_seq[::-1]
        self.loss_function = None
        self.classifier_loss_function = None
        self.classifier_function = None
        self.net_type = "Regression"
        self.paramters_dict = self.get_parameters()

    # 添加一个操作符 layer，顺序由调用者指定
    def add_op(self, operator):
        if len(self.operator_seq) == 0:
            operator.is_leaf_node = True
        else:
            operator.is_leaf_node = False
        self.operator_seq.append(operator)
        self.reversed_ops = self.operator_seq[::-1]
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
        if isinstance(combined_func, Classifier.LogisticCrossEntropy):
            self.set_classifier_function(Classifier.Logisitic())   # 二分类交叉熵损失函数
            self.set_loss_function(loss.CrossEntropy2())   # 二分类交叉熵损失函数
            self.net_type = "BinaryClassifier"
        if isinstance(combined_func, Classifier.SoftmaxCrossEntropy):
            self.set_classifier_function(Classifier.Softmax())   # 多分类交叉熵损失函数
            self.set_loss_function(loss.CrossEntropy3())   # 多分类交叉熵损失函数
            self.net_type = "Classifier"

    def forward(self, X):
        data = X
        for op in self.operator_seq:
            data = op.forward(data)
        if self.classifier_function is not None:
            data = self.classifier_function.forward(data)
        return data

    def predict(self, X):
        data = X
        for op in self.operator_seq:
            data = op.predict(data)
        if self.classifier_function is not None:
            data = self.classifier_function.forward(data)
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
        #predict = self.predict(x)  # 为啥用这个不行?
        predict = self.forward(x)
        loss = self.compute_loss(predict, label)
        if self.net_type == "Regression":
            accu = r2(label, loss)
        elif self.net_type == "BinaryClassifier":
            accu = tpn2(predict, label)
        elif self.net_type == "Classifier":
            accu = tpn3(predict, label)
        return loss, accu

    def testing(self, x, label):
        predict = self.predict(x)  # 为啥用这个不行?
        #predict = self.forward(x)
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

    # 父进程：更新权重参数
    def update_parameters_value(self, lr):
        for _, wb in self.paramters_dict.items():
            wb.Update(lr)

    # 父进程，只调用一次，用于共享参数和接收梯度
    def create_share_memory_for_training(self, num_process):
        self.shm_buf = {}  # 存储 name -> (shm,buf)
        for process_id in range(num_process):
            self.shm_buf[process_id] = {}  # 根据进程id建立二级字典
        for op_name, wb_obj in self.paramters_dict.items():
            WB_value = wb_obj.get_WB_value()
            # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
            wb_shm = shared_memory.SharedMemory(name=op_name, create=True, size=WB_value.nbytes)
            wb_buf = np.ndarray(WB_value.shape, dtype=WB_value.dtype, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)  # 放在一级字典中
            # 在父进程端建立，用于子进程共享 WB 梯度给父进程，每个子进程都需要有一份copy
            for process_id in range(num_process):
                grad_shm_name = str(process_id) + "_" + op_name + "_grad"
                grad_shm = shared_memory.SharedMemory(name=grad_shm_name, create=True, size=WB_value.nbytes)
                buf = np.ndarray(WB_value.shape, dtype=np.float32, buffer=grad_shm.buf)
                self.shm_buf[process_id][op_name] = (grad_shm, buf)

    # 父进程，只调用一次，用于共享参数和接收预测
    def create_share_memory_for_prediction(self, num_process, batch_pred):
        self.shm_buf = {}  # 存储 name -> (shm,buf)
        # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
        for op_name, wb_obj in self.paramters_dict.items():
            WB_value = wb_obj.get_WB_value()
            wb_shm = shared_memory.SharedMemory(name=op_name, create=True, size=WB_value.nbytes)
            wb_buf = np.ndarray(WB_value.shape, dtype=WB_value.dtype, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)  # 放在一级字典中
            print("main :", op_name, WB_value.shape)
        # 在父进程端建立，用于子进程共享预测结果给父进程
        for process_id in range(num_process):
            pred_shm_name = str(process_id) + "_pred"
            pred_shm = shared_memory.SharedMemory(name=pred_shm_name, create=True, size=batch_pred.nbytes)
            buf = np.ndarray(batch_pred.shape, dtype=np.float32, buffer=pred_shm.buf)
            self.shm_buf[pred_shm_name] = (pred_shm, buf)

    # 父进程：获得梯度数据共享，相加，除以进程数，赋值给主模型
    def get_grad_value(self, num_process):
        for op_name, wb_obj in self.paramters_dict.items():
            dWB = np.zeros_like(wb_obj.get_dWB_value())
            for process_id in range(num_process):
                # 累加从不同子进程传回来的梯度
                dWB += self.shm_buf[process_id][op_name][1][:]
                #print("main: get grad:", self.shm_buf[process_id][op_name][1][:].sum())
            # 取平均梯度
            dWB /= num_process
            wb_obj.set_dWB_value(dWB)

    # 父进程：获得预测结果共享
    def get_pred_value(self, num_process):
        results = []
        for process_id in range(num_process):
            pred_shm_name = str(process_id) + "_pred"
            batch_pred = self.shm_buf[pred_shm_name][1][:]
            results.append(batch_pred)
        return results

    # 父进程：把当前参数共享
    def share_parameters_value(self):
        for op_name, wb_obj in self.paramters_dict.items():
            value = wb_obj.get_WB_value()
            self.shm_buf[op_name][1][:] = value
            #print("main share WB:", value.sum())

    # 子进程，只调用一次，建立共享内存副本，用于接收参数和共享梯度
    def setup_share_memory(self, process_id):
        self.shm_buf = {}
        self.shm_buf[process_id] = {}
        for op_name, wb_obj in self.paramters_dict.items():
            WB = wb_obj.get_WB_value()
            # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
            wb_shm = shared_memory.SharedMemory(name=op_name, create=False, size=WB.nbytes)
            wb_buf = np.ndarray(WB.shape, dtype=np.float32, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)
            # 在父进程端建立的用于子进程共享 WB 梯度给父进程，每个子进程都需要有一份copy
            grad_shm_name = str(process_id) + "_" + op_name + "_grad"
            grad_shm = shared_memory.SharedMemory(name=grad_shm_name, create=False, size=WB.nbytes)
            grad_buf = np.ndarray(WB.shape, dtype=np.float32, buffer=grad_shm.buf)
            self.shm_buf[process_id][op_name] = (grad_shm, grad_buf)

    # 子进程，只调用一次，建立共享内存副本，用于接收参数和共享梯度
    def setup_share_memory_for_prediction(self, process_id, batch_pred):
        self.shm_buf = {}
        # 在父进程端建立的用于子进程共享预测结果给父进程
        pred_shm_name = str(process_id) + "_pred"
        pred_shm = shared_memory.SharedMemory(name=pred_shm_name, create=False, size=batch_pred.nbytes)
        pred_buf = np.ndarray(batch_pred.shape, dtype=np.float32, buffer=pred_shm.buf)
        self.shm_buf[pred_shm_name] = (pred_shm, pred_buf)
        # 在父进程端建立，父进程共享 WB 参数给子进程，只有一份copy即可
        for op_name, wb_obj in self.paramters_dict.items():
            WB = wb_obj.get_WB_value()
            wb_shm = shared_memory.SharedMemory(name=op_name, create=False, size=WB.nbytes)
            #print("--- ", op_name, WB.shape)
            wb_buf = np.ndarray(WB.shape, dtype=np.float32, buffer=wb_shm.buf)
            self.shm_buf[op_name] = (wb_shm, wb_buf)

    # 子进程：把梯度copy到share memory中
    def share_grad_value(self, process_id):
        for op_name, wb_obj in self.paramters_dict.items():
            value = wb_obj.get_dWB_value() # W 和 B 合在一起了
            self.shm_buf[process_id][op_name][1][:] = value
            #print("--- share grad value:", value.sum())
    
    # 子进程：把预测结果copy到share memory中
    def share_pred_value(self, process_id, batch_pred):
        pred_shm_name = str(process_id) + "_pred"
        self.shm_buf[pred_shm_name][1][:] = batch_pred

    # 子进程：从共享内存获得权重参数，赋值给本地模型
    def get_parameters_value(self):
        for op_name, wb_obj in self.paramters_dict.items():
            value = self.shm_buf[op_name][1][:]
            wb_obj.set_WB_value(value)
            #print("----get shared WB:", value.sum())

    def close_share_memory(self, num_process):
        for op_name, wb_obj in self.paramters_dict.items():
            self.shm_buf[op_name][0].close()
            self.shm_buf[op_name][0].unlink()
        # for process_id in range(num_process):
        #     pred_shm_name = str(process_id) + "_pred"
        #     self.shm_buf[pred_shm_name][0].close()
        #     self.shm_buf[pred_shm_name][0].unlink()

# 子进程信息，用于通信
class SubProcessInfo(object):
    def __init__(self, id, process, train_data_shm, train_data_buf, event_data, event_grad, event_update):
        self.id = id
        self.train_data_shm = train_data_shm
        self.train_data_buf = train_data_buf
        self.process = process
        self.event_data = event_data
        self.event_grad = event_grad
        self.event_update = event_update

    def share_train_data(self, data):
        self.train_data_buf[:] = data
        #print("main data size:", batch_data.sum())
        self.event_data.set()   # 通知子进程可以拿训练数据了
    
    def wait_grad_and_clear(self):
        self.event_grad.wait()   # 等待梯度数据
        self.event_grad.clear() # 得到梯度数据，清空标志

    def share_parameter_done(self):
        self.event_update.set()  # 通知子进程可以拿参数了
    
    def close(self):
        self.train_data_shm.close()
        self.train_data_shm.unlink()
        self.process.terminate()

# 子进程信息，用于通信
class SubProcessInfo_for_prediction(object):
    def __init__(self, id, process,
                 test_data_shm, test_data_buf,  
                 event_data, event_pred, event_update):
        self.id = id
        self.test_data_shm = test_data_shm
        self.test_data_buf = test_data_buf
        self.process = process
        self.event_data = event_data
        self.event_pred = event_pred
        self.event_update = event_update

    def share_test_data(self, data):
        self.test_data_buf[:] = data
        #print("main data size:", batch_data.sum())
        self.event_data.set()   # 通知子进程可以拿训练数据了
    
    def wait_pred_and_clear(self):
        self.event_pred.wait()   # 等待梯度数据
        self.event_pred.clear() # 得到梯度数据，清空标志

    def share_parameter_done(self):
        self.event_update.set()  # 通知子进程可以拿参数了
    
    def close(self):
        self.test_data_shm.close()
        self.test_data_shm.unlink()
        self.process.terminate()

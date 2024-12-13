from .OperatorBase import Operator
from . import Layers_7 as layer
from . import LossFunctions as loss
from .Estimators import r2, tpn2, tpn3

# 模型基类
class Module(object):
    # 目前这个operators必须有,否则从locals()中获取不到
    def save_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id) + ".txt"
            op.save(op_name)
            unique_id += 1

    def load_parameters(self, name, operators):
        unique_id = 0
        for op in operators:
            op_name = name + "_" + op.__class__.__name__ + "_" + str(unique_id) + ".txt"
            op.load(op_name)
            unique_id += 1


# 顺序计算类
class Sequential(Module):
    def __init__(self, *operators):
        self.operator_seq:list[Operator] = operators
        # for op in operators:
        #     print(op)
        self.reversed_ops = self.operator_seq[::-1]
        self.loss_function = None
        self.classifier_loss_function = None
        self.classifier_function = None
        self.net_type = "Regression"

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

    def update(self, lr):
        for op in self.operator_seq:
            op.update(lr)

    def compute_loss(self, predict, label):
        assert(self.loss_function is not None)
        return self.loss_function(predict, label)

    def get_regular_loss(self):
        regular_loss = 0
        for op in self.operator_seq:
            if isinstance(op, layer.Linear):
                regular_loss += op.get_regular_loss()
        return regular_loss

    def save(self, name):
        super().save_parameters(name, self.operator_seq)        

    def load(self, name):
        super().load_parameters(name, self.operator_seq)        

    def compute_loss_accuracy(self, x, label):
        predict = self.predict(x)
        loss = self.compute_loss(predict, label)
        if self.net_type == "Regression":
            accu = r2(label, loss)
        elif self.net_type == "BinaryClassifier":
            accu = tpn2(predict, label)
        elif self.net_type == "Classifier":
            accu = tpn3(predict, label)
        return loss, accu

    def forward_without_softmax(self, X):
        data = X
        for op in self.operator_seq:
            data = op.forward(data)
        return data

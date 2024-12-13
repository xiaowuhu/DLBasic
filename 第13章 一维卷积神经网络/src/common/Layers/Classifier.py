import numpy as np
from .Operator import Operator

# 二分类输出层
class Logisitic(Operator): # 等同于 Activators.Sigmoid
    def forward(self, z):
        self.z = z
        self.a = 1.0 / (1.0 + np.exp(-z))
        return self.a
    
    # 用 z 也可以计算，但是不如 a 方便
    def backward(self, delta_in):
        da = np.multiply(self.a, 1 - self.a) # 导数
        delta_out = np.multiply(delta_in, da)
        return delta_out

# 多分类输出层
class Softmax(Operator):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

    def backward(self, delta_in):
        pass

# 合并二分类输出和交叉熵损失函数
class LogisticCrossEntropy(Operator):
    def forward(self):
        pass
    
    # 联合求导
    def backward(self, predict, label):
        return predict - label

# 合并二分类输出和交叉熵损失函数
class SoftmaxCrossEntropy(Operator):
    def forward(self):
        pass
    
    # 联合求导
    def backward(self, predict, label):
        return predict - label

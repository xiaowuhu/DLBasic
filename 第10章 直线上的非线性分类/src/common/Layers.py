import numpy as np
import os
import sys
from .OperatorBase import Operator
from .WeightsBias import WeightsBias

# 线性映射层
class Linear(Operator):
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 init_method: str='normal',
                 optimizer: str="SGD",
    ): # W 是权重，B 是偏移
        self.WB = WeightsBias(input_size, output_size, init_method, optimizer)

    def get_WeightsBias(self):
        return self.WB

    def forward(self, input): # input 是输入
        self.input = input  # 保存输入，以便反向传播时使用
        return np.dot(input, self.WB.W) + self.WB.B

    def backward(self, delta_in): # delta 是反向传播的梯度
        m = self.input.shape[0]
        self.WB.dW = np.dot(self.input.T, delta_in) / m
        self.WB.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.WB.W.T)  # 传到前一层的梯度
        return delta_out
    
    # lr 可能需要全网统一调整，所以要当作参数从调用者传入
    def update(self, lr): # lr = learning rate 学习率
        self.WB.Update(lr)

    def load(self, name):
        WB = super().load_from_txt_file(name)
        self.WB.W = WB[0:-1].reshape(self.WB.W.shape)
        self.WB.B = WB[-1:].reshape(self.WB.B.shape)
        print(name, self.WB)
    
    def save(self, name):
        WB = np.concatenate((self.WB.W, self.WB.B))
        super().save_to_txt_file(name, WB)


# 二分类输出层
class Logistic(Operator): # 等同于 Activators.Sigmoid
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
        raise


# 合并二分类输出和交叉熵损失函数
class LogisticCrossEntropy(Operator):
    def forward(self):
        raise
    
    # 联合求导
    def backward(self, predict, label):
        return predict - label


class my_binary_classifier(Operator):
    def forward(self, z):
        self.z = z
        return 0.5 + np.arctan(z) / np.pi
    
    def backward(self, delta_in):
        da = 1 / (np.pi * (1 + np.square(self.z)))
        delta_out = np.multiply(da, delta_in)
        return delta_out

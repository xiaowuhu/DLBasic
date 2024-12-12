import numpy as np
import os
import sys

# 把所有的操作都看作是 Operator, 相当于PyTorch中的Layer
class Operator(object):
    def __init__(self):
        pass

    def forward(self, z):
        pass

    def backward(self, dz):
        pass

    def update(self):
        pass

    def load(self, name):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        return np.loadtxt(file_path)

    def save(self, name, value):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), name)
        np.savetxt(file_path, value, fmt="%f")

    # 正向：假设 x 为 (1X3), y 为 (4X3), 计算时 x 广播，z 为（4X3)
    # 反向：计算 x 分支的回传误差时，需要把 dz 按行累加
    def sum_derivation(self, input, delta_in, delta_out):
        # 两个必须都是数组，没有标量
        if isinstance(input, np.ndarray) and isinstance(delta_in, np.ndarray):
            # shape相同的话则不处理
            # shape不同的话，必须列数相同
            # 输出的尺寸比输入的尺寸大
            if input.shape != delta_in.shape and \
               input.shape[1] == delta_in.shape[1] and \
               input.shape[0] < delta_in.shape[0]: 
                # 传入的误差尺寸比输入的尺寸大，意味着输出的尺寸比输入的尺寸大，有广播
                # 所以需要把传入的误差按行相加，保留列的形状，作为输出
                delta_out_sum = np.sum(delta_out, axis=0, keepdims=True)
                return delta_out_sum
        return delta_out


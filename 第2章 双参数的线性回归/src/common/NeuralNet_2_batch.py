import numpy as np
import tqdm
import math
from .DataLoader_2 import DataLoader_2

# 神经网络类 2
class NeuralNet_2_batch(object):
    def __init__(self, dataLoader: DataLoader_2, w, b, lr=0.01, batch_size=10):
        self.data_loader = dataLoader
        self.w = w      # 权重值
        self.b = b      # 偏移值
        self.lr = lr    # 学习率
        self.batch_size = batch_size # 批大小

    # 前向计算
    def forward(self, X):
        return np.dot(X, self.w) + self.b  # 式（2.4.1）
    
    # 反向传播
    def backward(self, X, Y, Z):
        m = X.shape[0]
        dZ = Z - Y
        dw = np.dot(X.T, dZ) / m  # 式（1.5.3）
        db = np.mean(dZ, axis=0, keepdims=False) 
        self.w = self.w - self.lr * dw  # 式（2.6.4）
        self.b = self.b - self.lr * db  # 式（2.6.5）
    
    # 网络训练
    def train_test(self, epoch, checkpoint=1, add_start = False):
        self.W = []
        self.B = []
        if add_start is True:
            self.W.append(self.w)
            self.B.append(self.b)
        max_batch = math.ceil(self.data_loader.num_train / self.batch_size)
        batch_num = 0
        for i in tqdm.trange(epoch):
            self.data_loader.shuffle_data()
            for batch_id in range(max_batch):
                batch_X, batch_Y = self.data_loader.get_batch(self.batch_size, batch_id)
                batch_Z = self.forward(batch_X)
                self.backward(batch_X, batch_Y, batch_Z)
                batch_num += 1
                if batch_num % checkpoint == 0:
                    self.W.append(self.w)
                    self.B.append(self.b)

        return self.W, self.B
    
    # 网络推理
    def predict(self, x):
        return self.forward(x)

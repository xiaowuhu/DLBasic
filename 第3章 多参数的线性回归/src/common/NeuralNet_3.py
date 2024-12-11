import numpy as np
import tqdm
import math
from .DataLoader_3 import DataLoader_3

# 神经网络类 2
class NeuralNet_3(object):
    def __init__(self, dataLoader: DataLoader_3, W, B, lr=0.01, batch_size=10):
        self.data_loader = dataLoader
        self.W = W      # 权重值
        self.B = B      # 偏移值
        self.lr = lr    # 学习率
        self.batch_size = batch_size # 批大小

    # 前向计算
    def forward(self, X):
        return np.dot(X, self.W) + self.B  # 式（2.4.1）
    
    # 反向传播
    def backward(self, X, Y, Z):
        m = X.shape[0]
        dZ = Z - Y
        dw = np.dot(X.T, dZ) / m  # 式（1.5.3）
        # db = np.sum(dZ, axis=0, keepdims=False) / m
        db = np.mean(dZ, axis=0, keepdims=True)
        self.W = self.W - self.lr * dw  # 式（1.5.4）
        self.B = self.B - self.lr * db
    
    # 网络训练
    def train(self, epoch):
        batch_per_epoch = math.ceil(self.data_loader.num_sample / self.batch_size)
        for i in tqdm.trange(epoch):
            self.data_loader.shuffle_data()
            for batch_id in range(batch_per_epoch):
                batch_X, batch_Y = self.data_loader.get_batch(self.batch_size, batch_id)
                batch_Z = self.forward(batch_X)
                self.backward(batch_X, batch_Y, batch_Z)
    
    # 网络推理
    def predict(self, x, normalization=True):
        if normalization:
            x = self.data_loader.normalize_pred_data(x)
        y_pred = self.forward(x)
        if normalization:
            y_pred = self.data_loader.de_normalize_y_data(y_pred)
        return y_pred

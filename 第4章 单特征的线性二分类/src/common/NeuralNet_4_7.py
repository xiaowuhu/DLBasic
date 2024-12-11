import numpy as np
import math
from .DataLoader_4 import DataLoader_4
from .TrainingHistory_4 import TrainingHistory_4
from .Functions_4 import bce_loss, logistic, tpn

# 神经网络类 3，增加了 checkpoint 验证集
class NeuralNet_4_7(object):
    def __init__(self, dataLoader: DataLoader_4, W, B, lr=0.01, batch_size=10):
        self.data_loader = dataLoader
        self.W = W      # 权重值
        self.B = B      # 偏移值
        self.lr = lr    # 学习率
        self.batch_size = batch_size # 批大小

    # 前向计算
    def forward(self, X):
        Z = np.dot(X, self.W) + self.B
        A = logistic(Z)
        return A
    
    # 反向传播
    def backward(self, X, Y, A):
        m = X.shape[0]
        dZ = A - Y
        dw = np.dot(X.T, dZ) / m
        db = np.mean(dZ, axis=0, keepdims=True)
        if self.record:
            self.check_data.append((self.W[0,0], self.B[0,0], dw[0,0], db[0,0], X[0,0], Y[0,0]))
        self.W = self.W - self.lr * dw
        self.B = self.B - self.lr * db
    
    # 网络训练
    def train(self, epoch, checkpoint=100):
        self.check_data = []
        training_history = TrainingHistory_4(self.W, self.B)
        batch_per_epoch = math.ceil(self.data_loader.num_train / self.batch_size)
        iteration = 0
        self.record = False
        for i in range(epoch):
            self.data_loader.shuffle_data()
            for batch_id in range(batch_per_epoch):
                batch_X, batch_Y = self.data_loader.get_batch(self.batch_size, batch_id)
                batch_A = self.forward(batch_X)
                self.backward(batch_X, batch_Y, batch_A)
                iteration += 1
                if iteration == checkpoint:
                    self.record = True
                if iteration == checkpoint + 4:
                    self.record = False
                # if iteration % checkpoint == 0:
                #     train_loss, train_accu, val_loss, val_accu = self.checkpoint(training_history, iteration)
                #     print("Epoch %d, 训练集: loss %f, accu %f, 验证集: loss %f, accu %f" %(i, train_loss, train_accu, val_loss, val_accu))
        return self.check_data

    # 计算损失函数和准确率
    def checkpoint(self, training_history:TrainingHistory_4, iteration:int):
        # 训练集
        x, y = self.data_loader.get_train()
        a = self.forward(x)
        train_loss = bce_loss(a, y)
        train_accu = tpn(a, y)
        # 验证集
        x, y = self.data_loader.get_val()
        a = self.forward(x)
        val_loss = bce_loss(a, y)
        val_accu = tpn(a, y)
        # 记录历史        
        training_history.append(iteration, train_loss, train_accu, val_loss, val_accu, self.W, self.B)
        return train_loss, train_accu, val_loss, val_accu

    # 网络推理
    def predict(self, x, normalization=True):
        if normalization:
            x = self.data_loader.normalize_pred_data(x)
        y_pred = self.forward(x)
        if normalization:
            y_pred = self.data_loader.de_normalize_y_data(y_pred)
        return y_pred

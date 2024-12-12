import numpy as np
from tqdm import tqdm
import math
from .DataLoader_5 import DataLoader_5
from .TrainingHistory_5 import TrainingHistory_5
from .Functions_5 import bce_loss, logistic, tpn

# 神经网络类 3，增加了 checkpoint 验证集
class NeuralNet_5(object):
    def __init__(self, dataLoader: DataLoader_5, W, B, lr=0.01, batch_size=10):
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
        self.dw = np.dot(X.T, dZ) / m
        self.db = np.mean(dZ)

        self.w1_history.append(self.W[0, 0])
        self.w2_history.append(self.W[1, 0])
        self.w3_history.append(self.W[2, 0])
        self.b_history.append(self.B[0, 0])
        self.dw1_history.append(self.dw[0, 0])
        self.dw2_history.append(self.dw[1, 0])
        self.dw3_history.append(self.dw[2, 0])
        self.db_history.append(self.db)

        self.W = self.W - self.lr * self.dw
        self.B = self.B - self.lr * self.db
    
    # 网络训练
    def train(self, max_epoch, checkpoint=100):
        training_history = TrainingHistory_5(self.W, self.B)
        self.w1_history = []       
        self.w2_history = []
        self.w3_history = []
        self.b_history = []
        self.dw1_history = []       
        self.dw2_history = []
        self.dw3_history = []
        self.db_history = []
        batch_per_epoch = math.ceil(self.data_loader.num_train / self.batch_size)
        iteration = 0
        for epoch in range(max_epoch):
            if epoch == 50:
                print("epoch 50")
            self.data_loader.shuffle_data()
            for batch_id in range(batch_per_epoch):
                batch_X, batch_Y = self.data_loader.get_batch(self.batch_size, batch_id)
                batch_A = self.forward(batch_X)
                self.backward(batch_X, batch_Y, batch_A)
                iteration += 1
                if iteration % checkpoint == 0:
                    train_loss, train_accu, val_loss, val_accu = self.checkpoint(training_history, iteration)
                    print("轮数 %d, 迭代 %d, 训练集: loss %f, accu %f, 验证集: loss %f, accu %f"%(epoch, iteration, train_loss, train_accu, val_loss, val_accu))

        history = np.zeros((8, len(self.w1_history)))
        history[0] = self.w1_history
        history[1] = self.w2_history
        history[2] = self.w3_history
        history[3] = self.b_history
        history[4] = self.dw1_history
        history[5] = self.dw2_history
        history[6] = self.dw3_history
        history[7] = self.db_history


        return training_history, history

    # 计算损失函数和准确率
    def checkpoint(self, training_history:TrainingHistory_5, iteration:int):
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

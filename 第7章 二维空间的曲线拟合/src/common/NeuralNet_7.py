import numpy as np
from tqdm import tqdm
import math
from .DataLoader_7 import DataLoader_7
from .TrainingHistory_7 import TrainingHistory_7
from .Activators import Tanh, Sigmoid
from .Functions_7 import mse_loss, r2

class NeuralNet_7(object):
    def __init__(self, dataLoader: DataLoader_7, W1, B1, W2, B2, lr=0.01, batch_size=10):
        self.data_loader = dataLoader
        self.W1 = W1      # 权重值
        self.B1 = B1      # 偏移值
        self.W2 = W2      # 权重值
        self.B2 = B2      # 偏移值
        self.lr = lr    # 学习率
        self.batch_size = batch_size # 批大小
        self.activator = Tanh()

    # 前向计算
    def forward(self, X):
        # 第一层
        self.Z1 = np.dot(X, self.W1) + self.B1 # 式（7.4.1）
        self.A1 = self.activator.forward(self.Z1) # 式（7.4.2）
        # 第二层
        Z2 = np.dot(self.A1, self.W2) + self.B2 # 式（7.4.3）
        # print("Z1=",self.Z1)
        # print("A1=",self.A1)
        # print("Z2=",Z2)
        return Z2
    
    # 反向传播
    def backward(self, X, Y, Z):
        m = X.shape[0]
        # 第二层
        dZ2 = Z - Y # 式（7.4.5）
        dW2 = np.dot(self.A1.T, dZ2) / m # 式（7.4.6）
        dB2 = np.mean(dZ2, axis=0, keepdims=True) # 式（7.4.7）
        # 第一层
        dA1 = np.dot(dZ2, self.W2.T) # 式（7.4.8）
        dZ1 = np.multiply(dA1, self.activator.backward(self.Z1, self.A1)) # 式（7.4.9）
        dW1 = np.dot(X.T, dZ1) / m # 式（7.4.10）
        dB1 = np.mean(dZ1, axis=0, keepdims=True) # 式（7.4.11）
        # 更新权重
        self.W1 = self.W1 - self.lr * dW1
        self.B1 = self.B1 - self.lr * dB1
        self.W2 = self.W2 - self.lr * dW2
        self.B2 = self.B2 - self.lr * dB2
        # print("dZ2=",dZ2)
        # print("dA1=",dA1)
        # print("dZ1=",dZ1)
        # print("W1=", self.W1)
        # print("B1=", self.B1)
        # print("dW1=", self.dW1)
        # print("dB1=", self.dB1)
        # print("W2=", self.W2)
        # print("B2=", self.B2)
        # print("dW2=", self.dW2)
        # print("dB2=", self.dB2)
    
    # checkpoint 大于 1 时必须为整数，= epoch数
    # 小于 1 时可以为小数，如 0.1 表示10%的 epoch 数记录一次
    def train(self, max_epoch, checkpoint=1):
        training_history = TrainingHistory_7()
        batch_per_epoch = math.ceil(self.data_loader.num_train / self.batch_size)
        check_iteration = int(batch_per_epoch * checkpoint)
        iteration = 0
        for epoch in range(max_epoch):
            self.data_loader.shuffle_data()
            for batch_id in range(batch_per_epoch):
                batch_X, batch_Y = self.data_loader.get_batch(self.batch_size, batch_id)
                batch_A = self.forward(batch_X)
                self.backward(batch_X, batch_Y, batch_A)
                iteration += 1
                if iteration % check_iteration == 0:
                    train_loss, train_accu, val_loss, val_accu = self.check_loss_accu(training_history, iteration)
                    print("轮数 %d, 迭代 %d, 训练集: loss %f, accu %f, 验证集: loss %f, accu %f" %(epoch, iteration, train_loss, train_accu, val_loss, val_accu))
        return training_history

    # 计算损失函数和准确率
    def check_loss_accu(self, training_history:TrainingHistory_7, iteration:int):
        # 训练集
        x, y = self.data_loader.get_train()
        z = self.forward(x)
        train_loss = mse_loss(z, y)
        train_accu = r2(y, train_loss)
        # 验证集
        x, y = self.data_loader.get_val()
        if x is not None:
            z = self.forward(x)
            val_loss = mse_loss(z, y)
            val_accu = r2(y, val_loss)
        else:
            val_accu = train_accu
            val_loss = train_loss
        # 记录历史        
        training_history.append(iteration, train_loss, train_accu, val_loss, val_accu)
        return train_loss, train_accu, val_loss, val_accu

    # 网络推理
    def predict(self, x):  # x 必须是经过归一化的数据
        y_pred = self.forward(x)
        return y_pred  # 返回预测结果需要反归一化

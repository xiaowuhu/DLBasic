import os
import numpy as np

# 本类适用于第 2 章的数据和代码
class DataLoader_2(object):
    def __init__(self, file_path):
        self.file_path = file_path

    # 加载数据
    def load_data(self, dir_name=None):
        self.data = np.loadtxt(self.file_path)
        self.train_x = self.data[0]
        self.train_y = self.data[1]
        self.num_train = self.train_x.shape[0]
        return self.data
    
    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.data.shape[1])
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]
        return self.train_x, self.train_y
    
    # 归一化训练数据
    def normalize_train_data(self):
        self.x_min = np.min(self.data[0])
        self.x_max = np.max(self.data[0])
        # 式（2.5.1）
        self.train_x = (self.data[0] - self.x_min) / (self.x_max - self.x_min)
        self.y_min = np.min(self.data[1])
        self.y_max = np.max(self.data[1])
        # 式（2.5.2）
        self.train_y = (self.data[1,:] - self.y_min) / (self.y_max - self.y_min)

    # 归一化预测数据
    def normalize_pred_data(self, X):
        # 式（2.5.1）
        normlized_X = (X - self.x_min) / (self.x_max - self.x_min)
        return normlized_X
    
    # 反归一化预测结果
    def de_normalize_y_data(self, Y):
        # 式（2.5.3）
        denormalized_Y = Y * (self.y_max - self.y_min) + self.y_min
        return denormalized_Y
    
    # get batch training data
    def get_batch(self, batch_size, batch_id):
        start = batch_id * batch_size
        end = start + batch_size
        batch_X = self.train_x[start:end]
        batch_Y = self.train_y[start:end]
        return batch_X, batch_Y

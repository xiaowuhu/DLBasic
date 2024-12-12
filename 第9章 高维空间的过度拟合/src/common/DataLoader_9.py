import os
import numpy as np

# 本类适用于第 9 章的数据和代码
class DataLoader_9(object):
    def __init__(self, file_path_train, file_path_val):
        self.file_path_train = file_path_train
        self.file_path_val = file_path_val

    # 加载数据, 只加载指定的列号，最后一列是标签值
    def load_data(self, col_list: list = None):
        self.data = np.loadtxt(self.file_path_train)
        if col_list is not None:
            self.num_feature = len(col_list) - 1
            self.tmp_data = np.zeros((self.data.shape[0], len(col_list)))
            for i, col in enumerate(col_list):
                self.tmp_data[:, i] = self.data[:, col]
            self.data = self.tmp_data
        else:
            self.num_feature = self.data.shape[1] - 1
        self.train_x = self.data[:,0:self.num_feature]
        self.train_y = self.data[:,-1:]
        self.num_sample = self.train_x.shape[0]
        self.num_train = self.num_sample
        self.num_feature = self.train_x.shape[1]

        val_data = np.loadtxt(self.file_path_val)
        self.num_val = val_data.shape[0]
        self.val_x = val_data[:, 0:self.num_feature]
        self.val_y = val_data[:,-1:]
    
    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.num_train)
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]

 # =============================================

    # 0,1归一化训练数据
    def MinMaxScaler_X(self):
        self.x_min = np.min(self.train_x, axis=0)
        self.x_max = np.max(self.train_x, axis=0)
        self.train_x = (self.train_x - self.x_min) / (self.x_max - self.x_min)

    # 0,1 归一化预测数据x
    def MinMaxScaler_pred_X(self, X):
        normalized_x = (X - self.x_min) / (self.x_max - self.x_min)
        return normalized_x

    # 0,1反归一化预测数据
    def de_MinMaxScaler_X(self, X):
        de_normalized_X = X * (self.x_max - self.x_min) + self.x_min
        return de_normalized_X

    # =============================================
    
    # 0,1归一化标签数据
    def MinMaxScaler_Y(self):
        self.y_min = np.min(self.train_y, axis=0)
        self.y_max = np.max(self.train_y, axis=0)
        self.train_y = (self.train_y - self.y_min) / (self.y_max - self.y_min)

    # 0,1 反归一化预测结果
    def de_MinMaxScaler_Y(self, pred_Y):
        de_normalized_Y = pred_Y * (self.y_max - self.y_min) + self.y_min
        return de_normalized_Y

    # =============================================

    # 0 标准化训练数据
    def StandardScaler_X(self):
        self.x_mean = np.mean(self.train_x, axis=0)
        self.x_std = np.std(self.train_x, axis=0)
        self.train_x = (self.train_x - self.x_mean) / self.x_std

    # 0 标准化预测数据
    def StandardScaler_pred_X(self, X):
        normalized_x = (X - self.x_mean) / self.x_std
        return normalized_x

    # 0 反标准化预测数据
    def de_StandardScaler_X(self, X):
        de_normalized_X = X * self.x_std + self.x_mean
        return de_normalized_X

    # =============================================

    # 0 标准化标签数据
    def StandardScaler_Y(self):
        self.y_mean = np.mean(self.train_y, axis=0)
        self.y_std = np.std(self.train_y, axis=0)
        self.train_y = (self.train_y - self.y_mean) / self.y_std

    # 0 反标准化预测结果
    def de_StandardScaler_Y(self, pred_Y):
        de_normalized_Y = pred_Y * self.y_std + self.y_mean
        return de_normalized_Y

    # =============================================

    # 归一化验证数据
    def MinMaxScaler_val_XY(self):
        self.val_x = (self.val_x - self.x_min) / (self.x_max - self.x_min)
        self.val_y = (self.val_y - self.y_min) / (self.y_max - self.y_min)
   
    # 变成Onehot编码，在split_data之前调用本方法
    def to_onehot(self, num_classes):
        self.num_classes = num_classes
        self.train_y = np.eye(self.num_classes)[self.train_y.flatten()]

    # get batch training data
    def get_batch(self, batch_size, batch_id):
        start = batch_id * batch_size
        end = start + batch_size
        batch_X = self.train_x[start:end]
        batch_Y = self.train_y[start:end]
        return batch_X, batch_Y

    def get_val(self):
        return self.val_x, self.val_y

    def get_train(self):
        return self.train_x, self.train_y

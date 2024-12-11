import os
import numpy as np

# 本类适用于第 1 章的数据和代码
class DataLoader_1(object):
    def __init__(self, file_path):
        self.file_path = file_path
    # 加载数据
    def load_data(self, dir_name=None):
        self.data = np.loadtxt(self.file_path)
        return self.data
    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.data.shape[1])
        train_x = self.data[0][idx]
        train_y = self.data[1][idx]
        return train_x, train_y
    
    def get_data(self):
        return self.data[0], self.data[1] # x,y

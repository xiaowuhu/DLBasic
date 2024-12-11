import numpy as np

# 本类适用于第 3 章的数据和代码
class DataLoader_3(object):
    def __init__(self, file_path):
        self.file_path = file_path

    # 加载数据
    def load_data(self, dir_name=None):
        self.data = np.loadtxt(self.file_path)
        self.train_x = self.data[:,0:2]
        self.train_y = self.data[:,-1:]
        self.num_sample = self.train_x.shape[0]
        self.num_feature = self.train_x.shape[1]
        return self.data
    
    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.data.shape[0])
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]
        return self.train_x, self.train_y
    
    # 归一化训练数据
    def normalize_train_data(self):
        self.x_min = np.min(self.train_x, axis=0)
        self.x_max = np.max(self.train_x, axis=0)
        self.train_x = (self.train_x - self.x_min) / (self.x_max - self.x_min)
        self.y_min = np.min(self.train_y, axis=0)
        self.y_max = np.max(self.train_y, axis=0)
        self.train_y = (self.train_y - self.y_min) / (self.y_max - self.y_min)

    # 归一化预测数据
    def normalize_pred_data(self, X):
        normlized_X = (X - self.x_min) / (self.x_max - self.x_min)
        return normlized_X
    
    # 反归一化预测结果
    def de_normalize_y_data(self, Y):
        denormalized_Y = Y * (self.y_max - self.y_min) + self.y_min
        return denormalized_Y
    
    # get batch training data
    def get_batch(self, batch_size, batch_id):
        start = batch_id * batch_size
        end = start + batch_size
        batch_X = self.train_x[start:end]
        batch_Y = self.train_y[start:end]
        return batch_X, batch_Y

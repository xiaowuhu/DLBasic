import numpy as np

# 本类适用于第 4 章的数据和代码
class DataLoader_4(object):
    def __init__(self, file_path):
        self.file_path = file_path

    # 加载数据, 只加载指定的列号，最后一列是标签值
    def load_data(self, col_list: list):
        self.num_feature = len(col_list) - 1
        self.data = np.loadtxt(self.file_path)
        self.tmp_data = np.zeros((self.data.shape[0], len(col_list)))
        for i, col in enumerate(col_list):
            self.tmp_data[:, i] = self.data[:, col]
        self.data = self.tmp_data
        self.train_x = self.data[:,0:self.num_feature]
        self.train_y = self.data[:,-1:]
        self.num_sample = self.train_x.shape[0]
        self.num_train = self.num_sample
        self.num_val = 0
        self.num_feature = self.train_x.shape[1]
    
    # 分出验证集（但是需要先做归一化,然后打乱数据,再分验证集）
    def split_data(self, ratio=0.8):
        self.num_train = int(self.num_sample * ratio)
        # 验证集
        self.num_val = self.num_sample - self.num_train
        self.val_x = self.train_x[self.num_train:, :]
        self.val_y = self.train_y[self.num_train:, :]
        # 训练集
        self.train_x = self.train_x[:self.num_train, :]
        self.train_y = self.train_y[:self.num_train, :]

    # 打乱数据
    def shuffle_data(self):
        idx = np.random.permutation(self.num_train)
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]
    
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

    def get_val(self):
        return self.val_x, self.val_y

    def get_train(self):
        return self.train_x, self.train_y

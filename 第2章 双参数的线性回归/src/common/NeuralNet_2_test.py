import tqdm

# 神经网络类 2
class NeuralNet_2(object):
    def __init__(self, dataLoader, w, b, lr=0.01):
        self.data_loader = dataLoader
        self.w = w      # 权重值
        self.b = b      # 偏移值
        self.lr = lr    # 学习率
    # 前向计算
    def forward(self, x):
        return x * self.w + self.b  # 式（2.4.1）
    # 反向传播
    def backward(self, x, y, z):
        dz = z - y
        dw = dz * x  # 式（2.4.3）
        self.w = self.w - self.lr * dw  # 式（2.4.5）
        self.b = self.b - self.lr * dz
    # 网络训练
    def train(self, epoch):
        for i in tqdm.trange(epoch):
            train_x, train_y = self.data_loader.shuffle_data()
            for x, y in zip(train_x, train_y):
                z = self.forward(x)
                self.backward(x, y, z)
    # 网络推理
    def predict(self, x):
        return self.forward(x)

    def train_test(self, epoch, checkpoint=1000, add_start = False):
        self.W = []
        self.B = []
        if add_start is True:
            self.W.append(self.w)
            self.B.append(self.b)
        for i in tqdm.trange(epoch):
            if (i+1) % checkpoint == 0:
                self.W.append(self.w)
                self.B.append(self.b)
            train_x, train_y = self.data_loader.shuffle_data()
            for x, y in zip(train_x, train_y):
                z = self.forward(x)
                self.backward(x, y, z)
        return self.W, self.B

    def train_with_normalization(self, epoch):
        for i in tqdm.trange(epoch):

            train_x, train_y = self.data_loader.shuffle_data()
            for x, y in zip(train_x, train_y):
                z = self.forward(x)
                self.backward(x, y, z)

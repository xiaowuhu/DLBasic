
import tqdm

# 神经网络类 1
class NeuralNet_1(object):
    def __init__(self, dataLoader, w, lr=0.01):
        self.data_loader = dataLoader
        self.w = w      # 权重值
        self.lr = lr    # 学习率
    # 前向计算
    def forward(self, x):
        return x * self.w  # 式（1.5.1）
    # 反向传播
    def backward(self, x, y, z):
        dz = z - y
        dw = dz * x  # 式（1.5.3）
        self.w = self.w - self.lr * dw  # 式（1.5.4）
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


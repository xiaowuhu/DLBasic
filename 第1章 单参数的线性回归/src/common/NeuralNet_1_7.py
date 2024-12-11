
import tqdm

# 神经网络类 1
class NeuralNet_1_7(object):
    def __init__(self, dataLoader, w, lr=0.01):
        self.data_loader = dataLoader
        self.w = w      # 权重值
        self.lr = lr    # 学习率
        self.record = False

    # 前向计算
    def forward(self, x):
        return x * self.w  # 式（1.5.1）

    # 反向传播
    def backward(self, x, y, z):
        dz = z - y
        dw = dz * x  # 式（1.5.3）
        if self.record:
            self.data.append((self.w, dw, x, y, z, dz))
        self.w = self.w - self.lr * dw  # 式（1.5.4）

    # 网络训练
    def train(self, epoch, checkpoint = None):
        self.data = []
        iteration = 0
        for i in tqdm.trange(epoch):
            if checkpoint is not None and i == checkpoint:
                self.record = True

            train_x, train_y = self.data_loader.shuffle_data()
            for x, y in zip(train_x, train_y):
                if self.record:
                    iteration += 1
                    if iteration == 5:
                        self.record = False
                z = self.forward(x)
                self.backward(x, y, z)
        return self.data
    # 网络推理
    def predict(self, x):
        return self.forward(x)

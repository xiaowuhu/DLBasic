import numpy as np
from .OperatorBase import Operator
from .WeightsBias import WeightsBias

# 线性映射层
class Linear(Operator):
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 init_method: str='normal',
                 optimizer: str="SGD",
                 regularizer: tuple=("none", 0.0),
    ): # W 是权重，B 是偏移
        self.WB = WeightsBias(input_size, output_size, init_method, optimizer)
        if regularizer is not None:
            self.regular_name = regularizer[0]
            self.regular_value = regularizer[1]

    def get_WeightsBias(self):
        return self.WB

    def forward(self, input): # input 是输入
        self.input = input  # 保存输入，以便反向传播时使用
        return np.dot(input, self.WB.W) + self.WB.B

    def backward(self, delta_in): # delta 是反向传播的梯度
        m = self.input.shape[0]
        if self.regular_name == "L2":
            self.WB.dW = (np.dot(self.input.T, delta_in) + self.regular_value * self.WB.W) / m
            #self.WB.dW = np.dot(self.input.T, delta_in) / m + self.regular_value * self.WB.W
        elif self.regular_name == "L1":
            self.WB.dW = (np.dot(self.input.T, delta_in) + self.regular_value * np.sign(self.WB.W))/m
            # self.WB.dW = np.dot(self.input.T, delta_in) / m + self.regular_value * np.sign(self.WB.W)
        else:
            self.WB.dW = np.dot(self.input.T, delta_in) / m
        self.WB.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.WB.W.T)  # 传到前一层的梯度
        return delta_out
    
    # lr 可能需要全网统一调整，所以要当作参数从调用者传入
    def update(self, lr): # lr = learning rate 学习率
        self.WB.Update(lr)

    def get_regular_cost(self):
        if self.regular_name == "L1":
            return np.sum(np.abs(self.WB.W)) * self.regular_value
        elif self.regular_name == "L2":
            return np.sum(np.square(self.WB.W)) * self.regular_value
        else:
            return 0

    def load(self, name):
        WB = super().load(name)
        self.WB.W = WB[0:-1]
        self.WB.B = WB[-1:]
    
    def save(self, name):
        WB = np.concatenate((self.WB.W, self.WB.B))
        super().save(name, WB)


# 二分类输出层
class Logisitic(Operator): # 等同于 Activators.Sigmoid
    def forward(self, z):
        self.z = z
        self.a = 1.0 / (1.0 + np.exp(-z))
        return self.a
    
    # 用 z 也可以计算，但是不如 a 方便
    def backward(self, delta_in):
        da = np.multiply(self.a, 1 - self.a) # 导数
        delta_out = np.multiply(delta_in, da)
        return delta_out


# 多分类输出层
class Softmax(Operator):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

    def backward(self, delta_in):
        pass


# 丢弃层
class Dropout(Operator):
    def __init__(self, dropout_ratio=0.5):
        assert( 0 <= dropout_ratio <= 1)
        self.keep_ratio = 1 - dropout_ratio
        self.mask = None

    def forward(self, input):
        assert(input.ndim == 2)
        self.mask = np.random.binomial(n=1, p=self.keep_ratio, size=input.shape)
        self.z = input * self.mask / self.keep_ratio
        return self.z

    def predict(self, input):
        assert(input.ndim == 2)
        self.z = input * self.keep_ratio
        return self.z

    def backward(self, delta_in):
        delta_out = self.mask * delta_in / self.keep_ratio
        return delta_out
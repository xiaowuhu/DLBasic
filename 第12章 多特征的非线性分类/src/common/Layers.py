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
                 regularizer: tuple = ("None", 0.1),
    ): # W 是权重，B 是偏移
        self.WB = WeightsBias(input_size, output_size, init_method, optimizer)
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
        elif self.regular_name == "L1":
            self.WB.dW = (np.dot(self.input.T, delta_in) + self.regular_value * np.sign(self.WB.W))/m
        else:
            self.WB.dW = np.dot(self.input.T, delta_in) / m
        self.WB.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.WB.W.T)  # 传到前一层的梯度
        return delta_out
    
    # lr 可能需要全网统一调整，所以要当作参数从调用者传入
    def update(self, lr): # lr = learning rate 学习率
        self.WB.Update(lr)

    def get_regular_loss(self):
        if self.regular_name == "L1":
            return np.sum(np.abs(self.WB.W)) * self.regular_value
        elif self.regular_name == "L2":
            return np.sum(np.square(self.WB.W)) * self.regular_value
        else:
            return 0

    def load(self, name):
        WB = super().load_from_txt_file(name)
        self.WB.W = WB[0:-1].reshape(self.WB.W.shape)
        self.WB.B = WB[-1:].reshape(self.WB.B.shape)
        #print(name)
        #print("W=",self.WB.W)
        #print("B=",self.WB.B)
    
    def save(self, name):
        WB = np.concatenate((self.WB.W, self.WB.B))
        super().save_to_txt_file(name, WB)

    def get_parameters(self):
        WB = np.concatenate((self.WB.W, self.WB.B))
        return WB.shape

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

# 合并二分类输出和交叉熵损失函数
class LogisticCrossEntropy(Operator):
    def forward(self):
        pass
    
    # 联合求导
    def backward(self, predict, label):
        return predict - label


# 合并二分类输出和交叉熵损失函数
class SoftmaxCrossEntropy(Operator):
    def forward(self):
        pass
    
    # 联合求导
    def backward(self, predict, label):
        return predict - label


# 把输入的矩阵横向拼接
class Concat(Operator):
    def __init__(self, modules, input_size, output_size):
        self.modules = list(modules)
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        assert(len(self.modules) == len(self.input_size) == len(self.output_size))
        self.slice_input = []
        start_idx = 0
        for i in range(len(self.modules)):
            end_idx = start_idx + self.input_size[i]
            self.slice_input.append((start_idx, end_idx))
            start_idx = end_idx

        self.slice_output = []
        start_idx = 0
        for i in range(len(self.modules)):
            end_idx = start_idx + self.output_size[i]
            self.slice_output.append((start_idx, end_idx))
            start_idx = end_idx


    def forward(self, X, is_debug=False):
        outputs = []
        for i, module in enumerate(self.modules):
            output = module.forward(X[:, self.slice_input[i][0]:self.slice_input[i][1]], is_debug)
            outputs.append(output)
        output = np.hstack(outputs)
        return output

    def backward(self, delta_in):
        for i, module in enumerate(self.modules):
            # 由于前面没有其它网络，所以这个delta_out可丢弃
            delta_out = module.backward(delta_in[:, self.slice_output[i][0]:self.slice_output[i][1]])

    def update(self, lr): # lr = learning rate 学习率
        for module in self.modules:
            module.update(lr)

    def save(self, name):
        for module in self.modules:
            module.save(name)

    def load(self, name):
        for module in self.modules:
            module.load(name)

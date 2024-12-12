import numpy as np
from .OperatorBase import Operator

# 线性映射层
class Linear(Operator):
    def __init__(self, input_size, output_size): # W 是权重，B 是偏移
        self.W = np.random.randn(input_size, output_size)
        self.B = np.random.rand(1, output_size)
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)

    def forward(self, input): # input 是输入
        self.input = input  # 保存输入，以便反向传播时使用
        return np.dot(input, self.W) + self.B

    def backward(self, delta_in): # delta 是反向传播的梯度
        m = self.input.shape[0]
        self.dW = np.dot(self.input.T, delta_in) / m
        self.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.W.T)  # 传到前一层的梯度
        return delta_out
    
    # lr 可能需要全网统一调整，所以要当作参数从调用者传入
    def update(self, lr): # lr = learning rate 学习率
        self.W = self.W - lr * self.dW
        self.B = self.B - lr * self.dB

    def load(self, name):
        WB = super().load(name)
        self.W = WB[0:-1].reshape(self.W.shape)
        self.B = WB[-1:].reshape(self.B.shape)
        print(self.W, self.B)
    
    def save(self, name):
        WB = np.concatenate((self.W, self.B))
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


# 加法
class Add(Operator):
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x + y
        return z

    # dz/dx=1, dz/dy=1
    def backward(self, dz):
        dx = dz
        dy = dz
        dx = super().sum_derivation(self.x, dz, dx)
        dy = super().sum_derivation(self.y, dz, dy)
        if isinstance(self.x, np.ndarray) == True:
            assert(dx.shape == self.x.shape)
        if isinstance(self.y, np.ndarray) == True:
            assert(dy.shape == self.y.shape)
        return dx, dy

# 减法
class Sub(Operator):
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x - y
        return z
    
    def backward(self, dz):
        dx = dz
        dy = -dz
        dx = super().sum_derivation(self.x, dz, dx)
        dy = super().sum_derivation(self.y, dz, dy)
        assert(dx.shape == self.x.shape)
        assert(dy.shape == self.y.shape)
        return dx, dy
        #return 1, -1, dx, dy

class Sqrt(Operator):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        z = np.sqrt(x)
        return z

    # dz/dx = 1 / (2*sqrt(x))
    def backward(self, dz):
        dz_dx = 0.5 / np.sqrt(self.x)
        dx = dz_dx * dz
        dx = super().sum_derivation(self.x, dz, dx)
        return dx

class Mul(Operator):
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = self.x * self.y
        return z
    
    def backward(self, dz):
        dz_dx = self.y
        dz_dy = self.x
        dx = dz_dx * dz
        dy = dz_dy * dz
        dx = super().sum_derivation(self.x, dz, dx)
        dy = super().sum_derivation(self.y, dz, dy)
        assert(dx.shape == self.x.shape)
        assert(dy.shape == self.y.shape)
        return dx, dy
        #return dz_dx, dz_dy, dz_dx * dz, dz_dy * dz

class Div(Operator):
    def __call__(self, x, y):
        return self.forward(x, y)

    # y_i = A / D
    def forward(self, x, y):
        self.x = x
        self.y = y
        z = self.x / self.y
        return z

    # 检查 dz 的 shape 是否与 x,y 相等
    # 如果不相等，比如 (4,3) -> (4,1)，则需要做球和操作
    def backward(self, dz):
        dz_dx = 1 / self.y
        dz_dy = - self.x / (self.y * self.y)
        dx = dz_dx * dz
        dy = dz_dy * dz
        dx = super().sum_derivation(self.x, dz, dx)
        dy = super().sum_derivation(self.y, dz, dy)
        assert(dx.shape == self.x.shape)
        assert(dy.shape == self.y.shape)
        return dx, dy
        #return dz_dx, dz_dy, dz_dx * dz, dz_dy * dz

class Mean(Operator):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.m = x.shape[0]
        self.x = x
        z = np.mean(x, axis=0, keepdims=True)
        return z
    
    def backward(self, dz):
        dz_dx = 1 / self.m
        dx = dz * dz_dx
        dx = super().sum_derivation(self.x, dz, dx)
        return dx
        #return dz_dx, dz * dz_dx

class Square(Operator):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        z = x * x
        return z
    
    def backward(self, dz):
        dz_dx = 2 * self.x
        dx = dz_dx * dz
        dx = super().sum_derivation(self.x, dz, dx)
        return dx

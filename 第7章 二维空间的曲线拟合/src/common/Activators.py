
import numpy as np

# 激活函数
# 反向部分的输入应该为 z，但是有些函数利用正向计算出来的 a 值反哺可以方便计算

class Sigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z)) # 式（7.3.1)
        return a

    def backward(self, z, a):
        assert(a is not None)
        da = np.multiply(a, 1-a) # 式（7.3.2)
        return da

class Tanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0 # 式（7.3.3)
        return a

    def backward(self, z, a):
        assert(a is not None)
        da = 1 - np.multiply(a, a) # 式（7.3.4)
        return da

class Relu(object):
    def forward(self, z):
        a = np.maximum(0, z) # 式（7.3.5)
        return a
    # 注意这里判断的是输入时 z 的情况
    def backward(self, z, a):
        assert(z is not None)
        da = np.where(z > 0, 1, 0) # 式（7.3.6)
        return da

class LeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        a = np.maximum(0, z) + np.minimum(0, z) * self.alpha
        return a

    def backward(self, z, a):
        assert(z is not None)
        da = np.where(z >=0, 1, self.alpha)
        return da

class Elu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * (np.exp(x) - 1) for x in z]) # 式（7.3.9)

    def backward(self, z, a):
        assert(z is not None)
        da = np.array([1 if x > 0 else self.alpha * np.exp(x) for x in z]) # 式（7.3.10)
        return da

class Softplus(object):
    def forward(self, z):
        a = np.log(1 + np.exp(z)) # 式（7.3.11)
        return a

    def backward(self, z, a):
        assert(z is not None)
        p = np.exp(z) 
        da = p / (1 + p) # 式（7.3.12)
        return da

class GELU(object):
    def forward(self, z):
        # tmp1 = (z + 0.044715 * np.power(z,3)) * np.sqrt(2/np.pi)
        # tmp2 = Tanh().forward(tmp1)
        # tmp3 = 0.5 * z * (1 + tmp2)
        tmp1 = Sigmoid().forward(1.702 * z)
        tmp2 = z * tmp1
        return tmp2

    def backward(self, z, a):
        #return Sigmoid().forward(1.702*z) + z * Sigmoid().backward(1.702*z, a) * 1.702
        return Sigmoid().forward(1.702*z) + z * Sigmoid().forward(1.702*z) * (1 - Sigmoid().forward(1.702*z))

if __name__=="__main__":
    relu = Relu()
    x = np.array([1,-2,3,-4])
    a = relu.forward(x)
    print(x)
    print(a)
    print(relu.backward(x, a+2))


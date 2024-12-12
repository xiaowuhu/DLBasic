
import numpy as np

# 三分类函数
def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return a

if __name__ == "__main__":
    z = np.array([[-1, 0, 2]])
    a = softmax(z)
    print(a)

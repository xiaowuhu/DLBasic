import numpy as np
import os
import sys

# 把所有的操作都看作是 Operator, 相当于PyTorch中的Layer
class Operator(object):
    def load_from_txt_file(self, name):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name)
        return np.loadtxt(file_path)

    def save_to_txt_file(self, name, value):
        file_path = os.path.join(os.path.dirname(sys.argv[0]), "model", name)
        np.savetxt(file_path, value, fmt="%f")

    # 以下函数为那些没有相关操作的 op 给出缺省操作，如激活函数
    def update(self, lr):
        pass

    def load(self, name):
        pass

    def save(self, name):
        pass

    def predict(self, x):
        return self.forward(x)
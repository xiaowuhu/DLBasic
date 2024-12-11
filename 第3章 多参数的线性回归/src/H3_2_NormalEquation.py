
import numpy as np
import os
from common.DataLoader_3 import DataLoader_3

def get_current_path(name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filepath = os.path.join(current_dir, name)
    return filepath

def normal_equation(X, Y):
    # 在 X 前面加一列 1
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:])))
    # 开始计算正规方程
    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    return np.array(e)

if __name__ == '__main__':
    pred_x = np.array([[93, 15]])

    data = DataLoader_3(get_current_path("train3.txt"))
    data.load_data()
    Wb_t = normal_equation(data.train_x, data.train_y)
    print("-- 原始样本的正规方程结果 --")
    print("w_1 =", Wb_t[1, 0])
    print("w_2 =", Wb_t[2, 0])
    print("b =", Wb_t[0, 0])
    print("-- 用原始样本值预测 --")
    z = np.dot(pred_x, Wb_t[1:3]) + Wb_t[0]
    print("结果为", z)

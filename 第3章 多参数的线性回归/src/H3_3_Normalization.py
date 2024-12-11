
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

    print("-- 归一化样本的结果 --")
    data.normalize_train_data()
    print("X 的最小值", data.x_min)
    print("X 的最大值", data.x_max)
    print("X 的最大值 - 最小值 =", data.x_max - data.x_min)
    print("Y 的最小值", data.y_min)
    print("Y 的最大值", data.y_max)
    print("Y 的最大值 - 最小值 =", data.y_max - data.y_min)

    print("-- 归一化样本后的正规方程结果 --")
    Wb_n = normal_equation(data.train_x, data.train_y)
    print("w_1 =", Wb_n[1, 0])
    print("w_2 =", Wb_n[2, 0])
    print("b =", Wb_n[0, 0])

    print("-- 用归一化样本值预测 --")
    x = np.array([[93, 15]])
    x_new = data.normalize_pred_data(np.array([[93, 15]]))
    print("归一化样本特征值:", x, "=>", x_new)
    z = np.dot(x_new, Wb_n[1:3]) + Wb_n[0]
    y = data.de_normalize_y_data(z)
    print("归一化的预测结果:", z, "反归一化 =>", y)

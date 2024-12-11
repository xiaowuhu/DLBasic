
import os
from common.DataLoader_2 import DataLoader_2

# 式(2.3.15)
def method1(X, Y, m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

# 式(2.3.16)
def method2(X, Y):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

# 式(2.3.13)
def method3(X, Y, m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w

# 式(2.3.14)
def calculate_b_1(X, Y, w, m):
    b = sum(Y-w*X)/m
    return b

# 式(2.3.9)
def calculate_b_2(X, Y, w):
    b = Y.mean() - w * X.mean()
    return b

def calculate_w_b(X, Y):
    w = method2(X, Y)
    b = calculate_b_2(X, Y, w)
    return w, b

if __name__ == '__main__':
    print("---- 最小二乘法计算 train2.txt 中的 w,b ----")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()

    X,Y = data_loader.data[0], data_loader.data[1]
    m = X.shape[0]
    w1 = method1(X,Y,m)
    b1 = calculate_b_1(X,Y,w1,m)

    w2 = method2(X,Y)
    b2 = calculate_b_2(X,Y,w2)

    w3 = method3(X,Y,m)
    b3 = calculate_b_1(X,Y,w3,m)

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))


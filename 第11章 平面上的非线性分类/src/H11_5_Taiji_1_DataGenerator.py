import numpy as np
import os

# 三分类 同心圆
# 四分类

def generate_taiji_data(count):
    np.random.seed(4)
    r3 = 5  # 外圈大圆
    r32 = r3 * r3
    r2 = 2.5 # 上下两个中圆
    r22 = r2 * r2
    r1 = 1  # 上下两个小圆心
    r12 = r1 * r1
    Y = np.zeros((count, 1)) + 2
    X = np.random.uniform(low=-5, high=5, size=(count,2))
    for i in range(count):
        x = X[i, 0]
        y = X[i, 1]
        x2 = x * x
        y2 = y * y
        if (x2 + y2) < r32:  # 在大圆内
            if x2 + (y-2.5)**2 < r12: # 在上小圆内
                Y[i] = 0
            elif x2 + (y-2.5)**2 < r22: # 在上中圆内
                Y[i] = 1
            elif x2 + (y+2.5)**2 < r12: # 在下小圆内
                Y[i] = 0
            elif x2 + (y+2.5)**2 < r22: # 在下中圆内
                Y[i] = 3
            elif x < 0:  # 在左侧
                Y[i] = 1
            else:  # 在右侧
                Y[i] = 3
    
    # 删掉 Y=0 的点（在大圆的外面）
    Z = np.hstack((X,Y))
    X0 = Z[Y[:,0]==0]
    X1 = Z[Y[:,0]==1]
    X2 = Z[Y[:,0]==2]
    X3 = Z[Y[:,0]==3]
    X = np.vstack((X0, X1, X2, X3))

    return X


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir,  "data", name)
    np.savetxt(filename, data, fmt="%.4f", header="x,y,label")


if __name__ == '__main__':
    num_size = 1000
    train_data = generate_taiji_data(num_size)
    save_data(train_data, "train11-taiji.txt")

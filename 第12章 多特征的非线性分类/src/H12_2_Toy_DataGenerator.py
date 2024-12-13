import numpy as np
import os

# 生成数据
# 左侧 1，中侧 1，右侧 1
def mask1():
    np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1]
    ])
    np.random.choice(2, p=[1.0/3, 1.0/3, 1.0/3])
# 上方一，下方一，中间一
# 左下拐角    
# 右上拐角


def generate_img_noise_data(num):
    X = np.random.uniform(low=128, high=255, size=(num, 3, 3)).astype(np.int64)
    dict_classes = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2}
    mask = np.array([
        [[1, 0, 0], 
         [1, 0, 0],
         [1, 0, 0]],

        [[0, 1, 0],
         [0, 1, 0], 
         [0, 1, 0]],

        [[0, 0, 1], 
         [0, 0, 1], 
         [0, 0, 1]],

        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0], 
         [1, 1, 1], 
         [0, 0, 0]],

        [[0, 0, 0], 
         [0, 0, 0], 
         [1, 1, 1]],

        [[1, 0, 0],
         [1, 1, 0],
         [0, 0, 0]],

        [[0, 1, 0],
         [0, 1, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0]],

        [[0, 0, 0],
         [0, 1, 0],
         [0, 1, 1]]
    ])
    Y = np.zeros((num,1))
    for i in range(X.shape[0]):
        j = i % mask.shape[0]
        X[i] = X[i] * mask[j]
        Y[i] = dict_classes[j]
    
    return np.hstack((X.reshape(num,-1), Y))

def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.2f")

if __name__ == '__main__':
    data = generate_img_noise_data(100)
    save_data(data, "train12.txt")
    data = generate_img_noise_data(500)
    save_data(data, "train12_4.txt")

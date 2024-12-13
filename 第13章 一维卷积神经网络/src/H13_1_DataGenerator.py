import numpy as np
import os
import math

def generate_sin(x, count, y, noise):
    x = x * np.pi * 2
    X = np.sin(x)
    X = X + noise
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

def generate_sin_rev(x, count, y, noise):
    x = x * np.pi * 2
    X = np.sin(x)
    X = -1 * (X + noise)
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

def generate_cos(x, count, y, noise):
    x = x * np.pi * 2
    X = np.cos(x)
    X = X + noise
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

def generate_cos_rev(x, count, y, noise):
    x = x * np.pi * 2
    X = np.cos(x)
    X = -1 * (X + noise)
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

# 方波
def generate_square(x, count, y, noise):
    X = 0.5
    X = X + noise
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

def generate_square_rev(x, count, y, noise):
    X = 0.5
    X = -1 * (X + noise)
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

# 锯齿波
def generate_sawtooth(x, count, y, noise):
    X = [1,-1] * (len(x)//2)
    if len(x) % 2 != 0:
        X.append(1)
    X = X + noise
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))

def generate_sawtooth_rev(x, count, y, noise):
    X = [1,-1] * (len(x)//2)
    if len(x) % 2 != 0:
        X.append(1)
    X = -1 * (X + noise)
    Y = np.ones((count, 1)) * y
    return np.hstack((X, Y))


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.5f", header="x(1-9),y")

def generate_data(feature, count, name):
    x = np.linspace(0, 1, feature)
    noise = np.random.uniform(low=-0.4, high=0.4, size=(count,len(x)))
    #noise = np.random.uniform(low=-0.1, high=0.1, size=(count,len(x)))
    data1 = generate_sin(x, count, 0, noise)
    data2 = generate_cos(x, count, 1, noise)
    data3 = generate_sawtooth(x, count, 2, noise)
    data4 = generate_square(x, count, 3, noise)
    data5 = generate_sin_rev(x, count, 4, noise)
    data6 = generate_cos_rev(x, count, 5, noise)
    data7 = generate_sawtooth_rev(x, count, 6, noise)
    data8 = generate_square_rev(x, count, 7, noise)
    data = np.vstack((data1, data2, data3, data4, data5, data6, data7, data8))
    #data = np.vstack((data1, data2, data3))
    save_data(data, name)
    

if __name__=="__main__":
    np.random.seed(5)
    np.set_printoptions(suppress=True, precision=5)
    feature = 5
    generate_data(feature, 200, "train13.txt")
    generate_data(feature, 50, "test13.txt")

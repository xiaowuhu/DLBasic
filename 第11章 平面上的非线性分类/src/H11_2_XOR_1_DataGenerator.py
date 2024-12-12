import numpy as np
import os

def generate_xor_data(count):
    np.random.seed(5)
    scale = 0.7
    X00 = np.random.normal(loc=0, scale=scale, size=(count,1))
    Y00 = np.random.normal(loc=0, scale=scale, size=(count,1))
    X00 = np.hstack((X00 - 5 - np.min(X00), Y00 - 5 - np.min(Y00), np.zeros_like(X00)))

    X01 = np.random.normal(loc=0, scale=scale, size=(count,1))
    Y01 = np.random.normal(loc=0, scale=scale, size=(count,1))
    X01 = np.hstack((X01 - 5 - np.min(X01), Y01 + 5 - np.max(Y01), np.ones_like(X01)))

    X10 = np.random.normal(loc=0, scale=scale, size=(count,1))
    Y10 = np.random.normal(loc=0, scale=scale, size=(count,1))
    X10 = np.hstack((X10 + 5 - np.max(X10), Y10 - 5 - np.min(Y10), np.ones_like(X10)))

    X11 = np.random.normal(loc=0, scale=scale, size=(count,1))
    Y11 = np.random.normal(loc=0, scale=scale, size=(count,1))
    X11 = np.hstack((X11 + 5 - np.max(X11), Y11 + 5 - np.max(Y11), np.zeros_like(X11)))

    return np.vstack((X00, X01, X10, X11))


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savetxt(filename, data, fmt="%.4f", header="x,y,label")


if __name__ == '__main__':
    num_size = 100
    train_data = generate_xor_data(num_size)
    save_data(train_data, "train11-xor.txt")

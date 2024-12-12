import numpy as np
import os

def generate_moon_data(count):
    np.random.seed(6)
    # 上面的弯月
    x1 = np.linspace(-1, 1, count).reshape(count,1)
    noise = (np.random.random(count)-0.5)/4
    y1 = (np.cos(x1[:,0]) + noise).reshape(count,1) - 0.6
    X1 = np.hstack((x1,y1))
    Y1 = np.ones((count,1))

    # 下面的圆
    scale = 0.1
    x2 = np.random.normal(loc=0, scale=scale, size=(count,1))
    y2 = np.random.normal(loc=0, scale=scale, size=(count,1))
    X0 = np.hstack((x2, y2))    
    Y0 = np.zeros((count,1))

    X = np.concatenate((X0,X1))
    X = ((X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) - 0.5) * 10
    Y = np.concatenate((Y0,Y1))

    return np.hstack((X, Y))


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir,  "data", name)
    np.savetxt(filename, data, fmt="%.4f", header="x,y,label")


if __name__ == '__main__':
    num_size = 200
    train_data = generate_moon_data(num_size)
    save_data(train_data, "train11-moon.txt")

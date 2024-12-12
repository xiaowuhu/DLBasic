import numpy as np
import os

TRIANGLE_SIZE = 3
CIRCLE_SIZE = 4

def triangle_down(x):
    y = TRIANGLE_SIZE * (1 - np.sqrt(3))
    return y

def triangle_left(x):
    y = np.sqrt(3) * x + TRIANGLE_SIZE
    return y

def triangle_right(x):
    y = -np.sqrt(3) * x + TRIANGLE_SIZE
    return y

def in_triangle(x, y):
    if y > triangle_down(x) and y < triangle_left(x) and y < triangle_right(x):
        return True
    else:
        return False

def in_circle(x, y):
    r = np.sqrt(x*x + y*y)
    if r < CIRCLE_SIZE:
        return True
    else:
        return False

def in_square(x, y):
    if np.abs(x) < 2 and np.abs(y) < 2:
        return True
    return False


def generate_circle_data(count):
    X = np.random.uniform(low=-5, high=5, size=(count,2))
    Y = np.zeros((count,1))
    for i in range(count):
        x1 = X[i,0]
        x2 = X[i,1]
        t1 = in_circle(x1, x2)
        t2 = in_triangle(x1, x2)

        # 三分类
        if t2 == True:
            Y[i] = 2
        elif t1 == True and t2 == False:
            Y[i] = 1
    return np.hstack((X,Y))


def save_data(data, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir,  "data", name)
    np.savetxt(filename, data, fmt="%.4f", header="x,y,label")


if __name__ == '__main__':
    num_size = 1000
    train_data = generate_circle_data(num_size)
    save_data(train_data, "train11-stone.txt")

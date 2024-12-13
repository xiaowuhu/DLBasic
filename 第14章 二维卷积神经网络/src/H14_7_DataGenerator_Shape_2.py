import os
from PIL import Image, ImageDraw
import numpy as np

def select_center_radius():
    x = np.random.randint(4, 23)
    y = np.random.randint(4, 23)
    radius = np.random.randint(4,14)
    r = min(x, 27-x, y, 27-y, radius)
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return x0, y0, x1, y1

def circle(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    drawObj.ellipse([x0,y0,x1,y1], fill='white', outline='white')
    
def rectangle(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    drawObj.rectangle([x0,y0,x1,y1], fill='white', outline='white')

def save_data(name, data, label):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savez(filename, data=data, label=label)

def generate_circle(count, class_id):
    images = np.empty((count,1,28,28))
    for i in range(count):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        circle(drawObj)
        images[i,0] = np.array(img)
    
    labels = np.zeros((count, 1))
    labels.fill(class_id)
    return images, labels

def generate_rectangle(count, class_id):
    images = np.empty((count,1,28,28))
    for i in range(count):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        rectangle(drawObj)
        images[i,0] = np.array(img)
    labels = np.zeros((count, 1))
    labels.fill(class_id)
    return images, labels


def generate_data(count):
    x1, y1 = generate_circle(count, 0)
    x2, y2 = generate_rectangle(count, 1)
    x = np.vstack((x1, x2))
    y = np.vstack((y1, y2))
    return x, y

if __name__ == '__main__':
    np.random.seed(5)
    
    train_data_name = "train_shape_2.npz"
    test_data_name = "test_shape_2.npz"

    num_train = 1000
    x, y = generate_data(num_train)
    save_data(train_data_name, x, y)

    num_test = 200
    x, y = generate_data(num_test)
    save_data(test_data_name, x, y)

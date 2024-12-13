import os
from PIL import Image, ImageDraw
import numpy as np

def select_center_radius():
    # radius
    r = np.random.randint(6,13)
    # ccenter
    x = np.random.randint(r+1, 28-r-1)
    y = np.random.randint(r+1, 28-r-1)
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return x0, y0, x1, y1

def circle(drawObj, color):
    x0, y0, x1, y1 = select_center_radius()
    drawObj.ellipse([x0,y0,x1,y1], fill=color, outline=color)

def rectangle(drawObj, color):
    x0 = np.random.randint(1,12)
    y0 = np.random.randint(1,12)
    x1 = np.random.randint(15,26)
    y1 = np.random.randint(15,26)
    drawObj.rectangle([x0,y0,x1,y1], fill=color, outline=color)

def triangle(drawObj, color):
    x0 = np.random.randint(1,14)
    y0 = np.random.randint(1,14)
    x1 = np.random.randint(14,27)
    y1 = np.random.randint(1,14)
    x2 = np.random.randint(1,14)
    y2 = np.random.randint(14,27)
    x3 = np.random.randint(14,27)
    y3 = np.random.randint(14,27)
    r = np.random.randint(0,1)
    if r == 0:
        drawObj.polygon([x0,y0,x2,y2,x3,y3], fill=color, outline=color)
    else:
        drawObj.polygon([x1,y1,x2,y2,x3,y3], fill=color, outline=color)
    # elif r == 2:
    #     drawObj.polygon([x0,y0,x2,y2,x3,y3], fill='white', outline='white')
    # else:
    #     drawObj.polygon([x1,y1,x2,y2,x3,y3], fill='white', outline='white')


   
def generate_shape(count, class_id, func, color):
    images = np.empty((count,3,28,28))
    
    for i in range(count):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        func(drawObj, color)
        images[i] = np.array(img).transpose(2,0,1)
    
    labels = np.zeros((count, 1))
    labels.fill(class_id)
    return images, labels

def generate_data(count):
    colors = ["red", "green", "blue"]
    shapes = [circle, rectangle, triangle]

    X = []
    Y = []
    for i in range(3):
        color = colors[i]
        for j in range(3):
            func = shapes[j]
            label = i * 3 + j
            x, y = generate_shape(count, label, func, color)
            X.append(x)
            Y.append(y)
    
    x = np.vstack(X)
    y = np.vstack(Y)
    return x, y

def save_data(name, data, label):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    filename = os.path.join(current_dir, "data", name)
    np.savez(filename, data=data, label=label)

if __name__ == '__main__':
    np.random.seed(5)
    
    train_data_name = "train_shape_4.npz"
    test_data_name = "test_shape_4.npz"

    num_train = 500
    x, y = generate_data(num_train)
    save_data(train_data_name, x, y)

    num_test = 100
    x, y = generate_data(num_test)
    save_data(test_data_name, x, y)

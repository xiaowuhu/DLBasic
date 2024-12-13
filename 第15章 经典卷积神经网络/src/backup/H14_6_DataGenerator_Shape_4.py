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

def circle(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    drawObj.ellipse([x0,y0,x1,y1], fill='white', outline='white')

def half_circle_up(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    y2 = (y0 + y1)/2
    drawObj.pieslice([x0,y0,x1,y1], start=0,end=180,fill='white', outline='white')

def half_circle_down(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    y2 = (y0 + y1)/2
    drawObj.pieslice([x0,y0,x1,y1], start=180,end=0,fill='white', outline='white')

def half_circle_left(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    y2 = (y0 + y1)/2
    drawObj.pieslice([x0,y0,x1,y1], start=90,end=270,fill='white', outline='white')

def rectangle(drawObj):
    x0 = np.random.randint(1,12)
    y0 = np.random.randint(1,12)
    x1 = np.random.randint(15,26)
    y1 = np.random.randint(15,26)
    drawObj.rectangle([x0,y0,x1,y1], fill='white', outline='white')

# def rectangle(drawObj):
#     x0, y0, x1, y1 = select_center_radius()
#     drawObj.rectangle([x0,y0,x1,y1], fill='white', outline='white')

def diamond(drawObj):
    x0 = np.random.randint(1,12)
    y0 = np.random.randint(1,12)
    x1 = np.random.randint(15,26)
    y1 = np.random.randint(15,26)
    x2 = (x0+x1)/2
    y2 = (y0+y1)/2
    drawObj.polygon([x2,y0,x1,y2,x2,y1,x0,y2], fill='white', outline='white')

# def diamond(drawObj):
#     x0, y0, x1, y1 = select_center_radius()
#     x2 = (x0+x1)/2
#     y2 = (y0+y1)/2
#     drawObj.polygon([x2,y0,x1,y2,x2,y1,x0,y2], fill='white', outline='white')

def triangle(drawObj):
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
        drawObj.polygon([x0,y0,x2,y2,x3,y3], fill='white', outline='white')
    else:
        drawObj.polygon([x1,y1,x2,y2,x3,y3], fill='white', outline='white')
    # elif r == 2:
    #     drawObj.polygon([x0,y0,x2,y2,x3,y3], fill='white', outline='white')
    # else:
    #     drawObj.polygon([x1,y1,x2,y2,x3,y3], fill='white', outline='white')

# def triangle(drawObj):
#     x0, y0, x1, y1 = select_center_radius()
#     r = (x1 - x0) / 2
#     x2 = (x0 + x1) / 2
#     x3 = x2 - 1.732/2 * r
#     x4 = x2 + 1.732/2 * r
#     y2 = y0 + 3/2 * r
#     drawObj.polygon([x2,y0,x3,y2,x4,y2], fill='white', outline='white')

def hexagon(drawObj):
    x0, y0, x1, y1 = select_center_radius()
    r = (x1 - x0) / 2
    x2 = (x0 + x1) / 2
    x3 = x2 - 1.732/2 * r
    x4 = x2 + 1.732/2 * r
    y2 = y0 + 3/2 * r
    y3 = y2 - r
    drawObj.polygon([x2,y0,x3,y3,x3,y2,x2,y1,x4,y2,x4,y3], fill='white', outline='white')


def line(drawObj):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    x2 = np.random.randint(0,14)
    y2 = np.random.randint(14,28)
    x3 = np.random.randint(14,28)
    y3 = np.random.randint(0,14)
    r = np.random.randint(0,6)
    w = 3
    if r == 0:
        drawObj.line([x0,y0,x1,y1], fill='white', width=w)
    elif r == 1:
        drawObj.line([x0,y0,x2,y2], fill='white', width=w)
    elif r == 2:
        drawObj.line([x0,y0,x3,y3], fill='white', width=w)
    elif r == 3:
        drawObj.line([x1,y1,x2,y2], fill='white', width=w)
    elif r == 4:
        drawObj.line([x1,y1,x3,y3], fill='white', width=w)
    else: # r == 5:
        drawObj.line([x2,y2,x3,y3], fill='white', width=w)
   
def generate_shape(count, class_id, func):
    images = np.empty((count,1,28,28))
    for i in range(count):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        func(drawObj)
        images[i,0] = np.array(img)
    
    labels = np.zeros((count, 1))
    labels.fill(class_id)
    return images, labels

def generate_data(count):
    x1, y1 = generate_shape(count, 0, circle)
    x2, y2 = generate_shape(count, 1, half_circle_up)
    # x3, y3 = generate_shape(count, 2, half_circle_down)
    # x4, y4 = generate_shape(count, 3, half_circle_left)
    x3, y3 = generate_shape(count, 2, rectangle)
    x4, y4 = generate_shape(count, 3, diamond)
    x5, y5 = generate_shape(count, 4, triangle)
    #x5, y5 = generate_shape(count, 4, hexagon)
    x = np.vstack((x1, x2, x3, x4, x5))
    y = np.vstack((y1, y2, y3, y4, y5))
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

    num_train = 1000
    x, y = generate_data(num_train)
    save_data(train_data_name, x, y)

    num_test = 200
    x, y = generate_data(num_test)
    save_data(test_data_name, x, y)

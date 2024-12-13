import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 创建样本数据，从彩色图变成灰度图
def create_gray_image():
    circle_pic = os.path.join(os.path.dirname(__file__), "data", "circle.png")
    img = mpimg.imread(circle_pic)
    a = np.array([0.299, 0.587, 0.114, 0])
    gray = np.dot(img, a)
    return gray

# 计算卷积输出的尺寸
def calculate_output_size(input_h, input_w, kernel_h, kernel_w, padding, stride):
    output_h = (input_h + 2*padding - kernel_h) // stride + 1
    output_w = (input_w + 2*padding - kernel_w) // stride + 1
    return output_h, output_w

# 简单的二维卷积操作，传入output_buf节省每次创建的开销
def simple_conv2d(x, w, b, stride=1, padding=0):
    z = create_zero_array(x, w)
    input_h, input_w = x.shape
    kernel_h, kernel_w = w.shape
    out_h, out_w = calculate_output_size(input_h, input_w, kernel_h, kernel_w, 0, 1)
    for i in range(out_h):
        i_start = i * stride
        i_end = i_start + kernel_h
        for j in range(out_w):
            j_start = j * stride
            j_end = j_start + kernel_w
            z[i, j] = np.sum(x[i_start:i_end, j_start:j_end] * w)
    z += b
    return z

def create_zero_array(x,w):
    out_h, out_w = calculate_output_size(x.shape[0], x.shape[1], w.shape[0], w.shape[1], 0, 1)
    output = np.zeros((out_h, out_w))
    return output

def try_filters(img_gray):
    kernels = [
        np.array([0,-1,0,
                  -1,5,-1,
                  0,-1,0]),         # sharpness filter
        np.array([0,0,0,
                  -1,2,-1,
                  0,0,0]),          # vertical edge
        np.array([1,1,1,
                  1,-9,1,
                  1,1,1]),          # surround
        np.array([-1,-2,-1,
                  0,0,0,
                  1,2,1]),          # sobel y
        np.array([0,0,0,
                  0,1,0,
                  0,0,0]),          # nothing
        np.array([0,-1,0,
                  0,2,0,
                  0,-1,0]),         # horizontal edge
        np.array([0.11,0.11,0.11,
                  0.11,0.11,0.11,
                  0.11,0.11,0.11]), # blur
        np.array([-1,0,1,
                  -2,0,2,
                  -1,0,1]),         # sobel x
        np.array([2,0,0,
                  0,-1,0,
                  0,0,-1])]         # embossing

    filters_name = ["sharpness", "vertical edge", "surround", "sobel-y", "identity", "horizontal edge", "blur", "sobel-x", "embossing"]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
    for i in range(len(kernels)):
        z = simple_conv2d(img_gray, kernels[i].reshape(3,3), 0)
        ax[i//3, i%3].imshow(z, cmap='gray')
        ax[i//3, i%3].set_title(filters_name[i])
        ax[i//3, i%3].axis("off")
    plt.show()

if __name__ == '__main__':
    img_gray = create_gray_image()
    try_filters(img_gray)

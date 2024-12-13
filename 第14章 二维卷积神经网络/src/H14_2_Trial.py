import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 计算卷积输出的尺寸
def calculate_output_size(input_h, input_w, kernel_h, kernel_w, padding, stride):
    output_h = (input_h + 2*padding - kernel_h) // stride + 1
    output_w = (input_w + 2*padding - kernel_w) // stride + 1
    return output_h, output_w

# 简单的二维卷积操作，传入output_buf节省每次创建的开销
def simple_conv2d(x, w, b, output_buf, stride=1, padding=0):
    input_h, input_w = x.shape
    kernel_h, kernel_w = w.shape
    out_h, out_w = calculate_output_size(input_h, input_w, kernel_h, kernel_w, 0, 1)
    output_buf *= 0 # 使用前清 0
    for i in range(out_h):
        i_start = i * stride
        i_end = i_start + kernel_h
        for j in range(out_w):
            j_start = j * stride
            j_end = j_start + kernel_w
            output_buf[i, j] = np.sum(x[i_start:i_end, j_start:j_end] * w)
    output_buf += b
    return output_buf

def create_zero_array(x,w):
    out_h, out_w = calculate_output_size(x.shape[0], x.shape[1], w.shape[0], w.shape[1], 0, 1)
    output = np.zeros((out_h, out_w))
    return output

def train(x, w, b, y_label):
    z = create_zero_array(x, w)     # 输出缓存
    dw = np.zeros(w.shape)          # 梯度缓存
    lr = 0.5                        # 学习率

    for i in range(10000):
        z = simple_conv2d(x, w, b, z)   # 前向计算
        dz = z - y_label         # 梯度
        m = np.prod(dz.shape)    # 数据点数量作为批样本数
        loss = np.sum(np.multiply(dz, dz)) / 2 / m  # 计算损失
        if i % 100 == 0:            # 打印训练过程
            print(i,loss)
        if loss < 1e-7:             # 终止条件
            break
        dw = simple_conv2d(x, dz, b, dw)  # 反向传播
        w = w - lr * dw / m         # 梯度更新

    return w


def show_result(img_gray, w_true, w_result):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    y = create_zero_array(img_gray, w_true)
    simple_conv2d(img_gray, w_true, 0, y)
    ax[0].imshow(y, cmap='gray')
    ax[0].set_title("label")
    z = create_zero_array(img_gray, w_result)
    simple_conv2d(img_gray, w_result, 0, z)
    ax[1].imshow(z, cmap='gray')
    ax[1].set_title("predict")
    plt.show()

# 创建样本数据，从彩色图变成灰度图
def create_gray_image():
    circle_pic = os.path.join(os.path.dirname(__file__), "data", "circle.png")
    img = mpimg.imread(circle_pic)
    a = np.array([0.299, 0.587, 0.114, 0])
    gray = np.dot(img, a)
    return gray

# 卷积生成目标数据 y
def create_y_image(x):
    w = np.array([[0,-1,0],
                  [0, 2,0],
                  [0,-1,0]])
    b = 0
    y = create_zero_array(x, w)
    simple_conv2d(x, w, b, y)
    return w, y

# 对比显示两张图
def show_x_y(x, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    ax[0].imshow(x, cmap='gray')
    ax[0].set_title("x")
    ax[1].imshow(y, cmap='gray')
    ax[1].set_title("z")
    plt.show()    

if __name__ == '__main__':
    circle_pic = os.path.join(os.path.dirname(__file__), "data", "circle.png")
    x = create_gray_image()
    w_true, y_label = create_y_image(x)
    show_x_y(x, y_label)  

    # 随机初始化卷积核
    w_train = np.random.normal(0, 0.1, w_true.shape)
    # 训练
    w_train = train(x, w_train, 0, y_label)
    # 比较真实卷积核值和训练出来的卷积核值
    print("w_true:\n", w_true)
    print("w_train:\n", w_train)
    # 用训练出来的卷积核值对原始图片进行卷积
    y_pred = np.zeros(y_label.shape)
    simple_conv2d(x, w_true, 0, y_pred)
    # 与真实的卷积核的卷积结果比较
    show_result(x, w_true, w_train)
    # 比较卷积核值的差异核卷积结果的差异
    print("w allclose:", np.allclose(w_true, w_train, rtol=1e-2, atol=1e-2))
    print("y allclose:", np.allclose(y_label, y_pred, rtol=1e-2, atol=1e-2))

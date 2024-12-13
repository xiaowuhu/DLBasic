
import numpy as np
import common.Layers as layer
import time
import torch

def test_torch(count):
    conv_layer = torch.nn.MaxPool2d(kernel_height, stride=stride, padding=0)
    for i in range(count):
        x = torch.rand(batch_size, input_channels, data_height, data_width, requires_grad=True)
        output = conv_layer(x)
        delta = torch.rand(batch_size, input_channels, output_height, output_width)
        #output.backward(delta, retain_graph=True)
        output.backward(delta)

def test_simple(count):
    pool = layer.Pool2d((input_channels, data_height, data_width), (kernel_height, kernel_width), stride, 0)
    for i in range(count):
        x = np.random.rand(batch_size, input_channels, data_height, data_width)
        z1 = pool.forward_simple(x)
        dx1 = pool.backward_simple(dz)

def test_im2col(count):
    pool = layer.Pool2d((input_channels, data_height, data_width), (kernel_height, kernel_width), stride, 0)
    for i in range(count):
        x = np.random.rand(batch_size, input_channels, data_height, data_width)
        z1 = pool.forward_im2col(x)
        dx1 = pool.backward_col2im(dz)

def test(func, count):
    start = time.time()
    func(count)
    end = time.time()
    return end - start

if __name__=="__main__":
    np.set_printoptions(precision=4, suppress=True)
    input_channels = 3
    batch_size = 8
    stride = 2
    kernel_height = 2
    kernel_width = 2
    data_height = 7
    data_width = 6
    padding = 0

    output_height = 1 + (data_height + 2 * padding - kernel_height)//stride
    output_width = 1 + (data_width + 2 * padding - kernel_width)//stride

    test_torch(1)

    pool = layer.Pool2d((input_channels, data_height, data_width), (kernel_height, kernel_width), stride, 0)
    x = np.random.rand(batch_size, input_channels, data_height, data_width)
    z1 = pool.forward_simple(x)
    z2 = pool.forward_im2col(x)
    #print(z1)
    dz = z1 + 1
    dx1 = pool.backward_simple(dz)
    dx2 = pool.backward_col2im(dz)
    #print(dx1)
    print(np.allclose(z1, z2))
    print(np.allclose(dx1, dx2))

    print("计算三种种方法的1000次循环的时间...")

    count = 1000
    result = test(test_torch, count)
    print("--- 计算1000次前向+后向的耗时 ---")
    print("torch:", result)
    result = test(test_simple, count)
    print("simple:", result)
    result = test(test_im2col, count)
    print("im2col:", result)

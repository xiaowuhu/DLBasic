import torch
import numpy as np
import common.Layers as layer
import time

def test_torch(count):
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, (kernel_height, kernel_width), stride=stride)
    for i in range(count):
        x = torch.rand(batch_size, in_channels, data_height, data_width)
        output = conv_layer(x)
        delta = torch.rand(batch_size, out_channels, output_height, output_width)
        output.backward(delta)

def test_simple(count):
    conv = layer.Conv2d((in_channels, data_height, data_width), (out_channels, kernel_height, kernel_width), stride=stride)
    for i in range(count):
        x = np.random.rand(batch_size, in_channels, data_height, data_width)
        z = conv.forward_simple(x)
        dz = np.random.rand(batch_size, out_channels, output_height, output_width)
        conv.backward_transpose(dz)

def test_im2col(count):
    conv = layer.Conv2d((in_channels, data_height, data_width), (out_channels, kernel_height, kernel_width), stride=stride)
    for i in range(count):
        x = np.random.rand(batch_size, in_channels, data_height, data_width)
        z = conv.forward_im2col(x)
        dz = np.random.rand(batch_size, out_channels, output_height, output_width)
        conv.backward_col2im(dz)

def test(func, count):
    start = time.time()
    func(count)
    end = time.time()
    return end - start

if __name__=="__main__":
    np.set_printoptions(precision=4)
    in_channels = 3  #输入通道数量
    out_channels = 2 #输出通道数量
    data_height = 7     #每个输入通道上的卷积尺寸的高
    data_width = 6      #每个输入通道上的卷积尺寸的宽
    kernel_height = 4  #每个输入通道上的卷积尺寸
    kernel_width = 5  #每个输入通道上的卷积尺寸
    batch_size = 8   #批数量
    stride = 1
    padding = 0

    output_height = 1 + (data_height + 2 * padding - kernel_height)//stride
    output_width = 1 + (data_width + 2 * padding - kernel_width)//stride

    # warm-up
    count = 1
    test_torch(count)
    test_simple(count)
    test_im2col(count)

    count = 1000
    result = test(test_torch, count)
    print("--- 计算1000次前向+后向的耗时 ---")
    print("torch:", result)
    result = test(test_simple, count)
    print("simple:", result)
    result = test(test_im2col, count)
    print("im2col:", result)

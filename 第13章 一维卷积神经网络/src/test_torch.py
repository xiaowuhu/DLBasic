import torch
import numpy as np
import common.Layers as layer
import common.ConvLayer as conv_layer

def test_torch():
    in_channels = 1  #输入通道数量
    out_channels = 8 #输出通道数量
    width = 5      #每个输入通道上的卷积尺寸的宽
    heigth = 1     #每个输入通道上的卷积尺寸的高
    kernel_size = 5  #每个输入通道上的卷积尺寸
    batch_size = 6   #批数量
    stride = 1

    x = torch.rand(batch_size,in_channels,width)

#    x = torch.tensor([[1,2,3,4,5.0],[1,3,4,2,2],[1.1,2,3,4,5.0],[1,2,4,2,2]], requires_grad=True).reshape(batch_size,in_channels,width)
    conv_layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    out_put = conv_layer(x)
    print("----output---")
    print(out_put.shape)
    print(out_put)

    delta = torch.rand(batch_size, out_channels, 1)
    out_put.backward(delta)
    #print(x.grad)
    print("----weight---")
    print(conv_layer.weight.shape)
    print(conv_layer.weight)
    print("----weight grad---")
    print(conv_layer.weight.grad)
#    print("----bias shape---")
    # print(conv_layer.bias.shape)
    # print(conv_layer.bias)
    return x.numpy(), conv_layer.weight.detach().numpy(), delta.numpy() #, conv_layer.bias
    #y = torch.randn(8, 3, 4)
    #loss = torch.nn.MSELoss()


def test_my(x, weights, delta):
    in_channels = 1  #输入通道数量
    out_channels = 8 #输出通道数量
    width = 5      #每个输入通道上的卷积尺寸的宽
    heigth = 1     #每个输入通道上的卷积尺寸的高
    kernel_size = 5  #每个输入通道上的卷积尺寸
    batch_size = 6   #批数量
    stride = 1
    conv = layer.Conv1d((in_channels, width), (out_channels, kernel_size), stride=stride)
 #   x = np.array([[1.0,2,3,4.0,5],[1,3,4,2,2],[1,1,2,1,0]]).reshape(3,1,1,5)
 #   x = np.array([[1.0,2,3,4.0,5],[1,3,4,2,2],[1.1,2,3,4.0,5],[1,2,4,2,2]]).reshape(batch_size, in_channels, width)
    conv.WB.W = weights
    result = conv.forward(x)
    print("forward:\n", result)
    print(result.shape)
    result = conv.backward(delta)

    return

    #print("back:\n", result)
    #print(result.shape)
    print("------------")
    conv1 = conv_layer.ConvLayer((1, 1, 5), (3, 1, 4), (1, 0), ("normal","SDG", 0.1))
    conv1.initialize()
    conv1.set_filter(conv.WB.W, conv.WB.B)
    result = conv1.forward(x)
    print("forward:\n", result)
    print(result.shape)
    result = conv1.backward(delta_in, 0)
    print("back:\n", result)
    print(result.shape)

if __name__=="__main__":
    x, weights, delta= test_torch()
    test_my(x, weights, delta)


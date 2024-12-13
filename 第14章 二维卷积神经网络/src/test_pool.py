
import numpy as np
import common.Layers as layer
import torch


def test_torch_maxpool(kernel_size, stride, padding):
    x = torch.rand(batch_size, in_channels, data_height, data_width, requires_grad=True)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding)
    output = pool(x)
    dz = torch.rand(output.shape)
    output.backward(dz, retain_graph=True)
    return x.detach().numpy(), dz.numpy(), \
           output.detach().numpy(), x.grad.detach().numpy()

def test_my_maxpool(x, dz):
    pool = layer.Pool2d((in_channels, data_height, data_width), 
                 (kernel_size, kernel_size), stride=stride, pool_type="max")
    output = pool.forward(x)
    dx = pool.backward(dz)
    return output, dx

def test_torch_avgpool(kernel_size, stride, padding):
    x = torch.rand(batch_size, in_channels, data_height, data_width, requires_grad=True)
    pool = torch.nn.AvgPool2d(kernel_size, stride, padding)
    output = pool(x)
    dz = torch.rand(output.shape)
    output.backward(dz, retain_graph=True)
    return x.detach().numpy(), dz.numpy(), \
           output.detach().numpy(), x.grad.detach().numpy()

def test_my_avgpool(x, dz):
    pool = layer.Pool2d((in_channels, data_height, data_width), 
                 (kernel_size, kernel_size), stride=stride, pool_type="avg")
    output = pool.forward(x)
    dx = pool.backward(dz)
    return output, dx

if __name__=="__main__":
    in_channels = 3
    batch_size = 2
    stride = 1
    kernel_size = 2
    data_height = 5
    data_width = 5

    # x, dz, z0, dx0 = test_torch_maxpool(kernel_size, stride, 0)
    # z1, dx1 = test_my_maxpool(x, dz)
    # print(np.allclose(z0, z1))
    # print(np.allclose(dx0, dx1))

    x, dz, z0, dx0 = test_torch_avgpool(kernel_size, stride, 0)
    z1, dx1 = test_my_avgpool(x, dz)
    print(np.allclose(z0, z1))
    print(np.allclose(dx0, dx1))

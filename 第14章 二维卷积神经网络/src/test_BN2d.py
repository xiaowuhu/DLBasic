import common.Layers as layer
import numpy as np
import torch


def test_my_2d(x):
    bn = layer.BatchNorm2d(x.shape[1])
    z = bn.forward(x)
    dz = z + 1
    dx = bn.backward(dz)
    return z, dx, bn.WB.dW, bn.WB.dB


def test_torch_2d(batch, num_features, height, width):
    x = torch.randn(batch, num_features, height, width, requires_grad=True)
    bn = torch.nn.BatchNorm2d(num_features)
    z = bn(x)
    dz = z + 1
    z.backward(dz, retain_graph=True)
    return x.detach().numpy(), \
           z.detach().numpy(), \
           x.grad.detach().numpy(), \
           bn.weight.grad.detach().numpy(), \
           bn.bias.grad.detach().numpy()


def test_torch_from_scratch(x, num_features):
    gamma = torch.ones((1, num_features, 1, 1))
    beta = torch.zeros((1, num_features, 1, 1))
    mu = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    x_mu = x - mu
    var = (x_mu ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    y = x_mu / np.sqrt(var + 1e-5)
    z = gamma * y + beta
    return z

if __name__=="__main__":
    np.set_printoptions(precision=6, suppress=True)
    batch = 2    
    channel = 3
    height = 4
    width = 5
    x, z0, dx0, dw0, db0 = test_torch_2d(batch, channel, height, width)
    z1, dx1, dw1, db1 = test_my_2d(x)
    print("z", np.allclose(z0, z1, atol=1e-6, rtol=1e-7))
    print("dw", np.allclose(dw0.flatten(), dw1.flatten(), atol=1e-6, rtol=1e-7))
    print("db", np.allclose(db0, db1, atol=1e-6, rtol=1e-7))
    print("dx", np.allclose(dx0, dx1, atol=1e-6, rtol=1e-7))


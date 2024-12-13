import common.Layers_7 as layer
import numpy as np
import torch

def test_my(x):
    bn = layer.BatchNorm1d(x.shape[1])
    z = bn.forward(x)
    dz = x + 1
    dx = bn.backward(dz)
    return z, dx

def test_torch(batch, num_features):
    x = torch.randn(batch, num_features, requires_grad=True)
    bn = torch.nn.BatchNorm1d(num_features)
    z = bn(x)
    dz = x + 1
    z.backward(dz, retain_graph=True)
    return x.detach().numpy(), z.detach().numpy(), x.grad.detach().numpy()

if __name__=="__main__":
    num_features = 3
    batch = 4
    x, z, dx = test_torch(batch, num_features)
    z1, dx1 = test_my(x)
    print(np.allclose(z, z1, atol=1e-4, rtol=1e-5))
    print(np.allclose(dx, dx1, atol=1e-6, rtol=1e-7))

    print(x)
    print(dx)

import common.Layers as layer
import numpy as np
import torch

def test_my(x):
    bn = layer.BatchNorm1d(x.shape[1])
    z = bn.forward(x)
    dz = x + 1
    dx = bn.backward(dz)
    return z, dx, bn.WB.dW, bn.WB.dB

def test_torch(batch, num_features):
    x = torch.randn(batch, num_features, requires_grad=True)
    bn = torch.nn.BatchNorm1d(num_features)
    z = bn(x)
    dz = x + 1
    z.backward(dz, retain_graph=True)
    return x.detach().numpy(), z.detach().numpy(), x.grad.detach().numpy(), \
           bn.weight.grad.detach().numpy(), bn.bias.grad.detach().numpy(),

if __name__=="__main__":
    num_features = 3
    batch = 4
    x0, z0, dx0, dw0, db0 = test_torch(batch, num_features)
    z1, dx1, dw1, db1 = test_my(x0)
    print(np.allclose(z0, z1, atol=1e-6, rtol=1e-7))
    print(np.allclose(dx0, dx1, atol=1e-6, rtol=1e-7))
    print(np.allclose(dw0, dw1, atol=1e-6, rtol=1e-7))
    print(np.allclose(db0, db1, atol=1e-6, rtol=1e-7))

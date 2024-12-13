
import numpy as np
import common.Layers as layer
import torch


def test_torch_linear():
    x = torch.rand(batch_size, num_features, requires_grad=True)
    linear = torch.nn.Linear(num_features, num_features2)
    z = linear(x)
    relu = torch.nn.ReLU()
    a = relu(z)
    dz = torch.rand(batch_size, num_features2)
    a.backward(dz, retain_graph=True)
    return x.detach().numpy(), dz.numpy(), \
           a.detach().numpy(), x.grad.detach().numpy(), \
           linear.weight.detach().numpy(), linear.bias.detach().numpy(), \
           linear.weight.grad.detach().numpy(), linear.bias.grad.detach().numpy()

def test_my_linear(x, dz, w, b):
    linear = layer.Linear(num_features, num_features2)
    linear.WB.W = w.T
    linear.WB.B = b.reshape(linear.WB.B.shape)
    z = linear.forward(x)
    relu = layer.Relu()
    a = relu.forward(z)
    dz = relu.backward(dz)
    dx = linear.backward(dz)
    return a, dx, linear.WB.dW, linear.WB.dB

if __name__=="__main__":
    batch_size = 64
    num_features = 100
    num_features2 = 32

    x, dz, z0, dx0, w, b, dw0, db0 = test_torch_linear()
    z1, dx1, dw1, db1 = test_my_linear(x, dz, w, b)
    print(np.allclose(z0, z1, rtol=1e-7, atol=1e-6))
    print(np.allclose(dx0, dx1, rtol=1e-7, atol=1e-6))
    print(np.allclose(dw0/batch_size, dw1.T, rtol=1e-7, atol=1e-5))
    print(np.allclose(db0.reshape(db1.shape)/batch_size, db1, rtol=1e-7, atol=1e-5))

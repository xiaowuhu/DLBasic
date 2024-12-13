
import numpy as np
import common.Layers as layer
import torch


def test_torch_dropout(p):
    x = torch.rand(batch_size, data_width, requires_grad=True)
    dropout = torch.nn.Dropout(p)
    z = dropout(x)
    dz = z + 0.1
    z.backward(dz, retain_graph=True)
    return x.detach().numpy(), dz.detach().numpy(), \
           z.detach().numpy(), x.grad.detach().numpy()

def test_my_dropout(x, p):
    d = layer.Dropout(p)
    z = d.forward(x)
    dz = z + 0.1
    dx = d.backward(dz)
    return z, dx

if __name__=="__main__":
    batch_size = 1
    data_width = 10
    p = 0.1

    x, dz, z0, dx0 = test_torch_dropout(p)
    print(x)
    print("z0:", z0)
    print("dx0", dx0)
    z1, dx1 = test_my_dropout(x, p)
    print("z1:", z1)
    print("dx1:", dx1)
    print(np.allclose(z0, z1))
    print(np.allclose(dx0, dx1))

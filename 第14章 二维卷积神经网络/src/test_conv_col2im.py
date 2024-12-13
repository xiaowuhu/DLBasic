import torch
import numpy as np
import common.Layers as layer

def test_torch():
    x = torch.rand(batch_size, in_channels, data_height, data_width, requires_grad=True)
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, (kernel_height, kernel_width), stride=stride)
    output = conv_layer(x)
    delta = torch.rand(output.shape)
    output.backward(delta, retain_graph=True)
    #print(x.grad)
    # print("----weight---")
    # print(conv_layer.weight.shape)
    # print(conv_layer.weight)
    # print(conv_layer.bias)
    #print("----weight grad:", conv_layer.weight.grad.shape)
#    print("----bias shape---")
    # print(conv_layer.bias.shape)
#     # print(conv_layer.bias)
    return x.detach().numpy(), delta.numpy(), \
            output.detach().numpy(), \
            conv_layer.weight.detach().numpy(), \
            conv_layer.bias.detach().numpy(), \
            conv_layer.weight.grad.numpy(), \
            conv_layer.bias.grad.numpy(), \
            x.grad.detach().numpy()


def test_my(x, weights, bias, dz):
    conv = layer.Conv2d((in_channels, data_height, data_width), (out_channels, kernel_height, kernel_width), stride=stride)
    conv.WB.W = weights
    conv.WB.B = bias
    fw_im2col = conv.forward(x)
    dw_col2im, db_col2im = conv.backward_dw_col2im(dz)
    dx_col2im = conv.backward_dx_col2im(dz)

    return fw_im2col, dw_col2im, db_col2im, dx_col2im


def test():
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp+1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")    

if __name__=="__main__":
    #test()

    np.set_printoptions(precision=4)
    in_channels = 3  #输入通道数量
    out_channels = 4 #输出通道数量
    data_height = 28     #每个输入通道上的卷积尺寸的高
    data_width = 28      #每个输入通道上的卷积尺寸的宽
    kernel_height = 3  #每个输入通道上的卷积尺寸
    kernel_width = 3   #每个输入通道上的卷积尺寸
    batch_size = 64   #批数量
    stride = 2
    padding = 1

    x, dz, torch_output, weights, bias, dw_torch, db_torch, dx = test_torch()
    dw_torch /= batch_size
    db_torch /= batch_size
    fw_result, dw_result, db_result, dx_result = test_my(x, weights, bias, dz)
    print("fw: torch vs mine:", np.allclose(torch_output, fw_result, rtol=1e-5, atol=1e-5))
    print("dw: torch vs mine:", np.allclose(dw_torch, dw_result, rtol=1e-5, atol=1e-5))
    if np.allclose(dw_torch, dw_result, rtol=1e-5, atol=1e-5) == False:
        print("dw_torch:\n", dw_torch)
        print("dw_result:\n", dw_result)
    print("db: torch vs mine:", np.allclose(db_torch.flatten(), db_result.flatten(), rtol=1e-5, atol=1e-5))
    if np.allclose(db_torch.flatten(), db_result.flatten(), rtol=1e-5, atol=1e-5) == False:
        print("db_torch:\n", db_torch)
        print("db_result:\n", db_result)    
    print("dx: torch vs mine:", np.allclose(dx, dx_result, rtol=1e-5, atol=1e-5))



import torch
import torch.nn.functional as F

import common.Layers as layer



if __name__=="__main__":
    input = torch.tensor([[-1,-0.5,0],[-2,-1,0]], requires_grad=True, dtype=torch.float32)
    label = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
    target = input.softmax(dim=1)
    loss = F.cross_entropy(input, label)
    loss.backward(retain_graph=True)
    print("-----torch---------")
    print("loss=", loss)
    print("input=\n", input)
    print("predict=\n", target)
    print("grad=\n", input.grad)

    print("-----my---------")
    my_label = label.numpy()
    f = layer.SoftmaxCrossEntropy()
    my_loss, my_predict = f(input.detach().numpy(), my_label)
    print("loss=\n", my_loss)
    print("predict=\n", my_predict)
    grad = f.backward(my_predict, my_label)
    print("grad=\n", grad/2) # 批量=2

    celoss = torch.nn.CrossEntropyLoss()
    loss = celoss(input, label)
    print(loss)
    loss.backward(retain_graph=True)
    print(input.grad/2) # grad 累计了两次

import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

# 定义一个简单的神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, 8)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(8, 8)

    def forward2(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    def forward(self, x, a):
        x1 = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        if a == True:
            x = x + x1
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

if __name__ == "__main__":
    # 创建一个模型实例
    model = Net()
    # 确保模型处于训练模式
    model.train()
    # 定义一个用于检查梯度的回调函数
    grad_list = []
    def gradient_hook(module: nn.Module, grad_input, grad_output):
        print('------模型：', module)
        if len(grad_output) > 0 and grad_output[0] is not None:
            grad = torch.norm(grad_output[0])**2/grad_output[0].numel()
            grad_list.append(grad)
            print('输出梯度值模：', torch.norm(grad_output[0])**2/grad_output[0].numel(), grad_output[0].numel())
        if len(grad_input) > 0 and grad_input[0] is not None:
            print('输入梯度值模：', torch.norm(grad_input[0])**2/grad_input[0].numel(), grad_input[0].numel())

    # 遍历每个层，并注册回调函数
    for name, module in model.named_modules():
        module.register_full_backward_hook(gradient_hook)
    torch.manual_seed(0)
    # 创建一个输入张量
    x = torch.randn(1, 8)
    print("x:",x)
    y = torch.randn(1,8)
    print("y:",y)
    # 前向传播 有跳跃连接
    output1 = model(x, True)
    print("output:",output1)
    loss_function = nn.MSELoss()
    loss = loss_function(output1, y)
    # 反向传播
    loss.backward()
    grad_list1 = copy.deepcopy(grad_list)
    grad_list1[-2] += grad_list1[-1]
    grad_list1.pop(-1)
    grad_list.clear()
    
    print("---------------------------------------------------")
    # 前向传播 无跳跃连接
    output = model(x, False)
    print("output:",output)
    loss_function = nn.MSELoss()
    loss = loss_function(output, y)
    # 反向传播
    loss.backward()
    grad_list2 = copy.deepcopy(grad_list)

    print("grad_list1:",grad_list1)
    print("grad_list2:",grad_list2)

    plt.plot(grad_list1, label='有跳跃连接')
    plt.plot(grad_list2, label='无跳跃连接', linestyle='--')
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
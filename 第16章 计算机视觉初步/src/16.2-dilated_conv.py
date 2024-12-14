import torch
import torch.nn as nn

# 定义一个具有空洞卷积的简单卷积神经网络
class DilatedConvNet(nn.Module):
    def __init__(self):
        super(DilatedConvNet, self).__init__()
        # 定义一个空洞卷积层，kernel_size=3, dilation=2
        self.dilated_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        # 应用空洞卷积
        x = self.dilated_conv(x)
        return x

# 创建模型实例
model = DilatedConvNet()

# 创建一个随机输入张量，假设输入图片的大小为(1, 1, 28, 28)，即1个通道，28x28像素
input_tensor = torch.randn(1, 1, 28, 28)

# 将输入张量传递给模型
output = model(input_tensor)

print(output.shape)  # 输出张量的形状
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 三个不同膨胀率的3x3卷积
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        # 最后的 1x1 卷积，用于整合所有特征
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[2:]

        # 1x1 卷积
        x1 = self.relu(self.bn1(self.conv1x1(x)))

        # 三个不同膨胀率的3x3卷积
        x2 = self.relu(self.bn2(self.conv3x3_1(x)))
        x3 = self.relu(self.bn3(self.conv3x3_2(x)))
        x4 = self.relu(self.bn4(self.conv3x3_3(x)))

        # 全局平均池化
        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.bn5(self.conv1x1_pool(x5)))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)

        # 拼接所有特征
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # 最终的 1x1 卷积和 batch norm
        x = self.relu(self.bn_final(self.conv_final(x)))

        return x

# 实例化 ASPP 模块
aspp = ASPP(in_channels=256, out_channels=256, atrous_rates=[6, 12, 18])

# 创建一个输入 tensor，假设 batch size=1，通道数为256，尺寸为64x64
x = torch.randn(1, 256, 64, 64)

# 前向传播
output = aspp(x)
print(output.shape)  # 输出的尺寸应该是 torch.Size([1, 256, 64, 64])

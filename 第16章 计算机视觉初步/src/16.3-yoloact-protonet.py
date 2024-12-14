import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, in_channels=256, proto_out_channels=32):
        super(ProtoNet, self).__init__()
        
        # 定义5个卷积层，每个卷积层后跟随 ReLU 激活函数
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # 最后一个卷积层输出原型掩码
        self.conv5 = nn.Conv2d(256, proto_out_channels, kernel_size=3, padding=1)

        # 通过 bilinear 上采样恢复原始输入的空间分辨率
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 连续的卷积和 ReLU 操作
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 生成原型掩码
        x = self.conv5(x)
        
        # 上采样以恢复空间分辨率
        x = self.upsample(x)
        
        return x

protonet = ProtoNet()
print(protonet)
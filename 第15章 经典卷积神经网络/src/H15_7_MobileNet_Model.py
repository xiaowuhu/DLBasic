
import torch
import torch.nn as nn
import torch.nn.functional as F

# 可分离卷积模块
class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, strides=1):
        super(DepthwiseConv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=strides, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out

class MobileNet(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # should be stride=2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dp1_6 = nn.Sequential(
            DepthwiseConv(32, 64, strides=1),   # 1
            DepthwiseConv(64, 128, strides=1),  # 2, should be strides=2
            DepthwiseConv(128, 128, strides=1), # 3
            DepthwiseConv(128, 256, strides=2), # 4
            DepthwiseConv(256, 256, strides=1), # 5
            DepthwiseConv(256, 512, strides=2), # 6
        )
        self.dp7_11 = nn.Sequential(
            DepthwiseConv(512, 512, strides=1), # 7
            DepthwiseConv(512, 512, strides=1), # 8
            DepthwiseConv(512, 512, strides=1), # 9
            DepthwiseConv(512, 512, strides=1), # 10
            DepthwiseConv(512, 512, strides=1), # 11
        )
        self.dp12_13 = nn.Sequential(
            DepthwiseConv(512, 1024, strides=2),    # 12
            DepthwiseConv(1024, 1024, strides=1),   # 13
        )
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        out = self.input(x)
        out = self.dp1_6(out)
        out = self.dp7_11(out)
        out = self.dp12_13(out)
        out = self.output(out)
        return out


if __name__=="__main__":
    from torchsummary import summary
    model = MobileNet()
    # # 获得参数尺寸
    # print("---- paramaters size ----")
    # for name, param in model.named_parameters():
    #     print(f"Layer name:{name}, size: {param.size()}")
    # 获得每一层的输出尺寸
    # print("---- output size ----")
    # summary(model, (3,224,224))
    print(model)
    x = torch.rand(1, 3, 32, 32)
    for name, layer in model.named_children():
        x = layer(x)    
        print(name, "output shape:", x.shape)

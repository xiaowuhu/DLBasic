
import torch
import torch.nn as nn
import torch.nn.functional as F

# 旁路分支直连
class ResBlock_x(nn.Module):
    def __init__(self, in_ch, out_ch, strides=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        output1 = self.res_block(x)
        output = F.relu(output1 + x)
        return output

# 旁路分支有 1x1 卷积
class ResBlock_conv1x(nn.Module):
    def __init__(self, in_ch, out_ch, strides=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        output1 = self.res_block(x)
        output2 = self.shortcut(x)
        output = F.relu(output1 + output2)
        return output

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # using this to keep input size big
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # disable this to keep input size big
        )
        self.conv2_x = self._create_simple_block(2, 64, 64, 1)
        self.conv3_x = self._create_downsample_block(2, 64, 128, 2)
        self.conv4_x = self._create_downsample_block(2, 128, 256, 2)
        self.conv5_x = self._create_downsample_block(2, 256, 512, 2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def _create_simple_block(self, block_num, in_channels, out_channels, stride):
        blocks = []
        for i in range(block_num):
            blocks.append(ResBlock_x(in_channels, out_channels, strides=stride))
        return nn.Sequential(*blocks)

    def _create_downsample_block(self, block_num, in_channels, out_channels, stride):
        blocks = []
        for i in range(block_num):
            if i == 0:
                blocks.append(ResBlock_conv1x(in_channels, out_channels, strides=stride))
            else:
                blocks.append(ResBlock_x(out_channels, out_channels, strides=stride))
        return nn.Sequential(*blocks)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.classifier(output)
        return output


if __name__=="__main__":
    from torchsummary import summary
    model = ResNet18()
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
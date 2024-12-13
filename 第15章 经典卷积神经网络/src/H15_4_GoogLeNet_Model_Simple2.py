import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax

class Inception(nn.Module):
    def __init__(self, in_ch, c1_out, c2_out, c3_out, c4_out):
        super(Inception, self).__init__()
        # 1 x 1
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, c1_out, kernel_size=1),
            nn.BatchNorm2d(c1_out),
            nn.ReLU(inplace=True),
        )
        # 1 x 1 -> 3 x 3
        self.c2 = nn.Sequential(
            nn.Conv2d(in_ch, c2_out[0], kernel_size=1),
            nn.BatchNorm2d(c2_out[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2_out[0], c2_out[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2_out[1]),
            nn.ReLU(inplace=True),
        )
        # 1 x 1 -> 5 x 5
        self.c3 = nn.Sequential(
            nn.Conv2d(in_ch, c3_out[0], kernel_size=1),
            nn.BatchNorm2d(c3_out[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_out[0], c3_out[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3_out[1]),
            nn.ReLU(inplace=True),
        )        
        # P 3 x 3 -> C 1 x 1
        self.c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, c4_out, kernel_size=1),
            nn.BatchNorm2d(c4_out),
            nn.ReLU(inplace=True),
        )        
    
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        cat = torch.cat((x1, x2, x3, x4), dim=1)
        return cat

class InceptionOutput(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(InceptionOutput, self).__init__()
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_ch, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 32),  # nn.Linear(2048, 1024)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes), # nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        output = self.output(x)
        return output

class GoogLeNet_Simple2(nn.Module):
    def __init__(self, in_ch, num_classes, aux_output = True):
        super(GoogLeNet_Simple2, self).__init__()
        self.aux_output = aux_output  # 是否需要额外的两个输出
        self.input = nn.Sequential(
            # 3 x 224 x 224
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            # 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # 64 x 56 x 56
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 56 x 56
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # 192 x 56 x 56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.part1 = nn.Sequential(
            # 192 x 28 x 28
            # 3a
            Inception(192, c1_out=64, c2_out=[96,128], c3_out=[16,32], c4_out=32),
            # 256 x 28 x 28
            # 3b
            Inception(256, c1_out=128, c2_out=[128,192], c3_out=[32,96], c4_out=64),
            # 480 x 28 x 28
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # 480 x 14 x 14
            # 4a
            Inception(480, c1_out=192, c2_out=[96,208], c3_out=[16,48], c4_out=64),
        )
        self.output1 = InceptionOutput(512, num_classes)
        self.part2 = nn.Sequential(
            # 512 x 14 x 14
            # 4b
            Inception(512, c1_out=160, c2_out=[112,224], c3_out=[24,64], c4_out=64),
            # 512 x 14 x 14
            # 4c
            Inception(512, c1_out=128, c2_out=[128,256], c3_out=[24,64], c4_out=64),
            # 512 x 14 x 14
            # 4d
            Inception(512, c1_out=112, c2_out=[144,288], c3_out=[32,64], c4_out=64),
        )
        self.output2 = InceptionOutput(528, num_classes)
    
    def forward(self, x):
        x = self.input(x)
        x = self.part1(x)
        if self.training:
            output1 = self.output1(x)
        x = self.part2(x)
        output2 = self.output2(x)
        if self.training:
            return output1, output2
        else:
            return output2



if __name__=="__main__":
    from torchsummary import summary
    model = GoogLeNet_Simple2(3, 10).cuda()
    # # 获得参数尺寸
    # print("---- paramaters size ----")
    # for name, param in model.named_parameters():
    #     print(f"Layer name:{name}, size: {param.size()}")
    # 获得每一层的输出尺寸
    print("---- output size ----")
    summary(model, (3,32,32))
    
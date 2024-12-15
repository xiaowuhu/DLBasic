import os
import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim  # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
import torchvision.transforms.v2 as transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import time
from torch.utils.data import TensorDataset, DataLoader

from H15_2_Train_Base import train_model, eval_model, load_model
from H15_6_ResNet_Model import ResNet18

def create_data_loader(batch_size):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # 下面的值使用 H15_5_Cifar100_ShowData.py 计算得到
                transforms.Normalize((0.5071, 0.4856, 0.4409), (0.2673, 0.2564, 0.2762)),
                # 数据增强，在训练后期使用
                transforms.RandomHorizontalFlip(p=0.1),
                #transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=(-10,10)),
                transforms.RandomAffine(0, translate=(0.1,0.1)),
                #transforms.RandomPerspective(p=0.1),
                transforms.RandomErasing(p=0.1),
                transforms.RandomCrop(32, padding=4),
            ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # 下面的值使用 H15_5_Cifar100_ShowData.py 计算得到
                transforms.Normalize((0.5088, 0.4874, 0.4419), (0.2683, 0.2574, 0.2771)),
        ])),
        batch_size=batch_size)
    
    return train_loader, test_loader


def main(net:nn.Module, save_name:str, pretrained_model_name:str = None):
    batch_size = 64
    epoch = 100
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")  # 不想用 GPU 时（你疯了吧）
    train_loader, test_loader = create_data_loader(batch_size)
    # 初始化模型 将网络操作移动到GPU或者CPU
    model:nn.Module = net.to(DEVICE)
    #print(model)

    #加载预训练模型
    if pretrained_model_name is not None:
        load_model(model, pretrained_model_name, DEVICE)

    # 定义交叉熵损失函数
    #loss_func = nn.CrossEntropyLoss().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    # 定义模型优化器：输入模型参数，定义初始学习率
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80, 100], gamma=0.2) #learning rate decay
    best = 0
    train_model(epoch, model, DEVICE, train_loader, test_loader, optimizer, train_scheduler, loss_func, save_name, best_correct=best)


if __name__=="__main__":
    start = time.time()    
    net = ResNet18()
    #print(net)
    #main(net, "ResNet18_Cifar100.pth", pretrained_model_name="ResNet18_Cifar100_7543.pth")
    train_loader, test_loader = create_data_loader(64)
    eval_model(net, test_loader, "ResNet18_Cifar100_7563.pth")
    end = time.time()
    print("time:", end-start)

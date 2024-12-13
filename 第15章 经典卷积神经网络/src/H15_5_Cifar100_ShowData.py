import os
import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim  # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import time
from torch.utils.data import TensorDataset, DataLoader


def create_data_loader(batch_size):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.51, 0.48, 0.44), (0.27, 0.25, 0.28)),
                # 数据增强，在训练后期使用
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                # transforms.RandomAffine(0, translate=(0.1,0.1)),
                # transforms.RandomCrop(32, padding=4),
            ])),
        batch_size=batch_size, shuffle=True)  # 指明批量大小，打乱，这是后续训练的需要。
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.51, 0.48, 0.44), (0.27, 0.25, 0.28)),
        ])),
        batch_size=batch_size)
    
    return train_loader, test_loader

# 用全量数据计算均值和标准差，当数据集很大时会很慢，甚至内存不够
def compute_mean_std_whole(data_loader):
    train = [item[0] for item in data_loader]
    x = torch.concat(train)
    assert(x.dim() == 4)
    mean = torch.mean(x, axis=(0,2,3))
    std = torch.std(x, axis=(0,2,3))
    print(mean, std)
    return mean, std

# 用批量数据计算计算均值和标准差
def compute_mean_std_batch(data_loader):
    sample_count = 0
    rgb_sum_m = torch.zeros(3)
    rgb_sum_s = torch.zeros(3)
    sample_count = len(data_loader.dataset)
    for x, _ in data_loader:
        rgb_sum_m += torch.sum(x, axis=(0,2,3))
        rgb_sum_s += torch.sum(torch.square(x), axis=(0,2,3))
    total_count = sample_count * x.shape[2] * x.shape[3]        
    rgb_mean = rgb_sum_m / total_count
    DX = rgb_sum_s / total_count - rgb_mean * rgb_mean
    rgb_std = torch.sqrt(DX)
    print("样本数量:", sample_count)
    return rgb_mean, rgb_std

if __name__=="__main__":
    batch_size = 1000
    print("reading data...")
    train_loader, test_loader = create_data_loader(batch_size)
    print("computing...")
    mean, std = compute_mean_std_batch(train_loader)
    print("训练集均值和标准差:\n", mean, std)
    # compute_mean_std_whole(train_loader)
    print("computing...")
    mean, std = compute_mean_std_batch(test_loader)
    print("测试集均值和标准差:\n", mean, std)
    #compute_mean_std_whole(test_loader)

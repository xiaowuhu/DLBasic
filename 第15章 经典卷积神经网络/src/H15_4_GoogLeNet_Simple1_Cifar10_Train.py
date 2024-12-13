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

from H15_2_Train_Base import test_model, load_model, print_training_progress, print_test_progress, save_model, eval_model
from H15_4_GoogLeNet_Model_Simple1 import GoogLeNet_Simple1

def create_data_loader(batch_size):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # 下面的值使用 H15_3_Cifar10_ShowData.py 计算得到
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                # 数据增强
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.RandomRotation(degrees=(0,15)),
                transforms.RandomAffine(degrees=0, translate=(0.2,0.2)),
                transforms.RandomCrop(32, padding=4),
                #transforms.RandomPerspective(distortion_scale=0.6, p=0.4),
                #transforms.ElasticTransform(alpha=50.),
            ])),
        batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # 下面的值使用 H15_3_Cifar10_ShowData.py 计算得到
                transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2467, 0.2429, 0.2616)),
        ])),
        batch_size=batch_size)
    
    return train_loader, test_loader

# 由于 loss 部分特殊，所以单独实现
def train_model(num_epochs, model, device, train_loader, test_loader, optimizer, lr_scheduler, loss_func, name):
    best_correct = 6720  # 自定义初始 best 便于及时保存结果，否则从 0 开始自动增长
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        # 从迭代器抽取图片和标签
        for step, (train_x, train_y) in enumerate(train_loader):
            x = train_x.to(device)
            y = train_y.to(device)
            # 此时样本是一批图片，在CNN的输入中，我们需要将其变为四维，
            # reshape第一个-1 代表自动计算批量图片的数目n
            # 最后reshape得到的结果就是n张图片，每一张图片都是单通道的28 * 28，得到四维张量
            predict = model(x)
            # 计算损失函数值
            step_loss = loss_func(predict, y)
            running_loss += step_loss
            # 优化器内部参数梯度必须变为0
            optimizer.zero_grad()
            # 损失值后向传播
            step_loss.backward()
            # 更新模型参数
            optimizer.step()
            print_training_progress(epoch, num_epochs, step, len(train_loader), step_loss, lr_scheduler)
        lr_scheduler.step()
        print()
        # 显示不在整数倍的最后一批数据
        test_loss, correct = test_model(test_loader, model, device, loss_func)
        print_test_progress(running_loss/len(train_loader), len(test_loader.dataset), test_loss, correct)
        if correct > best_correct:
            save_model(model, name)
            best_correct = correct

def main(net:nn.Module, save_name:str, pretrained_model_name:str = None):
    batch_size = 128
    epoch = 100
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")  # 不想用 GPU 时（你疯了吧）
    train_loader, test_loader = create_data_loader(batch_size)
    # 初始化模型 将网络操作移动到GPU或者CPU
    model:nn.Module = net.to(DEVICE)
    print(model)

    #加载预训练模型
    if pretrained_model_name is not None:
        load_model(model, pretrained_model_name, DEVICE)

    # 定义交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()  # default reduction = "mean"
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.0001, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001, weight_decay=1e-6)
    # 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    train_model(epoch, model, DEVICE, train_loader, test_loader, optimizer, exp_lr_scheduler, loss_func, save_name)

if __name__=="__main__":
    start = time.time()    
    net = GoogLeNet_Simple1(3, 10)
    #main(net, "GoogLeNet_Simple_Cifar10.pth", pretrained_model_name="GoogLeNet_Simple_Cifar10_6711.pth")
    train_loader, test_loader = create_data_loader(64)
    eval_model(net, test_loader, "GoogLeNet_Simple1_Cifar10_6968.pth")
    end = time.time()
    print("time:", end-start)

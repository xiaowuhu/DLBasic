import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import time

#torch.backends.cudnn.enabled = False

# 设计模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 提取特征层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(400, 120), 
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84), 
            nn.ReLU(inplace=True),
            nn.Linear(84, 26), 
        )
    # 前向传递函数    
    def forward(self, x):
        # 经过特征提取层
        x = self.features(x)
        # 输出结果必须展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def check_loss(_model, loss_func, epoch, num_epochs, i, loss_train, _lr_scheduler):
    # val_output = _model(val_x)
    # loss_val = loss_func(val_output, val_y)
    # idx = val_output.argmax(axis=1)
    # accu = val_y.eq(idx).sum() / val_y.shape[0]
    print("Epoch:{}/{}, step:{}, loss_train:{:.4f}, lr:{:.4f}".format(
        epoch + 1, num_epochs, i + 1, loss_train.item(),
        _lr_scheduler.get_last_lr()[0]))

def train(num_epochs, _model, _device, _train_loader, _test_loader, _optimizer, _lr_scheduler, loss_func):
    for epoch in range(num_epochs):
        _model.train()  # 设置模型为训练模式
        # 从迭代器抽取图片和标签
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)-1
            # 此时样本是一批图片，在CNN的输入中，我们需要将其变为四维，
            # reshape第一个-1 代表自动计算批量图片的数目n
            # 最后reshape得到的结果就是n张图片，每一张图片都是单通道的28 * 28，得到四维张量
            output = _model(samples)
            # 计算损失函数值
            loss_train = loss_func(output, labels)
            # 优化器内部参数梯度必须变为0
            _optimizer.zero_grad()
            # 损失值后向传播
            loss_train.backward()
            # 更新模型参数
            _optimizer.step()
            if (i + 1) % 100 == 0:
                check_loss(_model, loss_func, epoch, num_epochs, i, loss_train, _lr_scheduler)
            _lr_scheduler.step()
        test(_test_loader, _model, _device, loss_func)

def test(_test_loader, _model, _device, loss_func):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
 
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        count = 0
        for _data, _target in _test_loader:
            count += 1
            x, y = _data.to(_device), _target.to(_device)-1
            output = _model(x.reshape(-1, 1, 28, 28))
            loss += loss_func(output, y).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
 
    loss /= count
 
    print('测试集: \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))

def main():
    # 预设网络超参数 （所谓超参数就是可以人为设定的参数   
    BATCH_SIZE = 128  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
    EPOCHS = 10  # 总共训练迭代的次数
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    learning_rate = 0.01  # 设定初始的学习
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('data', train=True, download=True, split="letters",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.131,), std=(0.308,))  # 数据规范化到正态分布
            ])),
        batch_size=BATCH_SIZE, shuffle=True)  # 指明批量大小，打乱，这是后续训练的需要。
    
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('data', train=False, split="letters",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.132,), (0.310,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型 将网络操作移动到GPU或者CPU
    ConvModel = ConvNet().to(DEVICE)

    print(ConvModel)

    # 定义交叉熵损失函数
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    # 定义模型优化器：输入模型参数，定义初始学习率
    optimizer = torch.optim.Adam(ConvModel.parameters(), lr=learning_rate)
    # 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    train(EPOCHS, ConvModel, DEVICE, train_loader, test_loader, optimizer, exp_lr_scheduler, loss_func)

    #torch.save(ConvModel.state_dict(), "d:/Gitee/nnbasic/深度学习基础/第14章 二维卷积网络/src/model/ConvModel_LeNet.pth")
    #test(ConvModel, test_loader,ConvModel, DEVICE, loss_func)

if __name__=="__main__":
    start = time.time()    
    main()
    end = time.time()
    print("time:", end-start)

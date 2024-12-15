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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            # 单层
            nn.Linear(512, 10),
            # 三层
            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 10),
            # 双层
            # nn.Linear(512, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            # nn.Linear(32, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


def check_loss(_model, loss_func, epoch, num_epochs, i, loss_train, _lr_scheduler):
    # val_output = _model(val_x)
    # loss_val = loss_func(val_output, val_y)
    # idx = val_output.argmax(axis=1)
    # accu = val_y.eq(idx).sum() / val_y.shape[0]
    print("Epoch:{}/{}, step:{}, loss_train:{:.4f}, lr:{:.4f}".format(
        epoch + 1, num_epochs, i + 1, loss_train.item(),
        _lr_scheduler.get_last_lr()[0]))

def train(num_epochs, _model, _device, _train_loader, _test_loader, _optimizer, _lr_scheduler, loss_func):
    best_correct = 0  # 自定义初始 best 便于及时保存结果，否则从 0 开始自动增长
    for epoch in range(num_epochs):
        _model.train()  # 设置模型为训练模式
        # 从迭代器抽取图片和标签
        count = 0
        for i, (images, labels) in enumerate(_train_loader):
            count += images.shape[0]
            samples = images.to(_device)
            labels = labels.to(_device)
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
        print(count)
        # 显示不在整数倍的最后一批数据
        check_loss(_model, loss_func, epoch, num_epochs, i, loss_train, _lr_scheduler)
        correct = test(_test_loader, _model, _device, loss_func)
        if correct > best_correct:
            save_model(_model)
            best_correct = correct

def test(_test_loader, _model, _device, loss_func):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        count = 0
        for _data, _target in _test_loader:
            count += 1
            x, y = _data.to(_device), _target.to(_device)
            output = _model(x.reshape(-1, 3, 32, 32))
            loss += loss_func(output, y).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
 
    loss /= count
    print('测试集: \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))
    return correct

def save_model(model):
    print("---- save model... ----\n")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", "ConvModel.pth")
    torch.save(model.state_dict(), train_pth)

def raw_data_loader(BATCH_SIZE):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # 数据增强，在训练后期使用
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                # transforms.RandomAffine(0, translate=(0.1,0.1)),
                # transforms.RandomCrop(32, padding=4),
            ])),
        batch_size=BATCH_SIZE, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=BATCH_SIZE)
    return train_loader, test_loader

def load_mode(vgg, name):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    vgg.load_state_dict(torch.load(model_pth))

def main():
    # 预设网络超参数 （所谓超参数就是可以人为设定的参数   
    BATCH_SIZE = 128  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
    EPOCHS = 100  # 总共训练迭代的次数
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")  # 不想用 GPU 时可以用（你疯了吧）

    train_loader, test_loader = raw_data_loader(BATCH_SIZE)
    # 初始化模型 将网络操作移动到GPU或者CPU
    vgg = VGG("VGG13").to(DEVICE)
    print(vgg)

    #加载预训练模型
    #load_mode(vgg, "ConvModel_VGG13_1_8958.pth")

    # 定义交叉熵损失函数
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    # 定义模型优化器：输入模型参数，定义初始学习率
    optimizer = torch.optim.SGD(vgg.parameters(), momentum=0.9, nesterov=True, lr=0.01, weight_decay=1e-6)
    # 使用SGD做fine-tune
    #optimizer = torch.optim.Adam(vgg.parameters(), lr= 0.01, weight_decay=1e-6)
    # 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
    train(EPOCHS, vgg, DEVICE, train_loader, test_loader, optimizer, exp_lr_scheduler, loss_func)


def eval_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = VGG("VGG13").to(DEVICE)
    load_mode(vgg, "ConvModel_VGG_9198.pth")
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    train_loader, test_loader = raw_data_loader(100)
    test(test_loader, vgg, DEVICE, loss_func)


if __name__=="__main__":
    start = time.time()    
    main()
    eval_model()    
    end = time.time()
    print("time:", end-start)

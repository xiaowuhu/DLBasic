import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim  # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
from torch.utils.data import TensorDataset, DataLoader
import time
import os
from common.DataLoader_14 import DataLoader_14

# 设计模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 提取特征层
        conv1_out = 2   # 2
        conv2_out = 4  # 16
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv1_out, kernel_size=3, stride=1, padding=0),  # -> 26
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1, padding=0),   #  -> 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->12
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(conv2_out * 12 * 12, 9), 
        )
    # 前向传递函数    
    def forward(self, x):
        # 经过特征提取层
        x = self.features(x)
        # 输出结果必须展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def check_loss(_model, val_x, val_y, loss_func, epoch, num_epochs, i, loss_train, _lr_scheduler):
    val_output = _model(val_x)
    loss_val = loss_func(val_output, val_y)
    idx = val_output.argmax(axis=1)
    accu = val_y.eq(idx).sum() / val_y.shape[0]
    print("Epoch:{}/{}, step:{}, loss_train:{:.4f}, loss_val:{:.4f}, accu_val:{:.4f}, lr:{:.4f}".format(
        epoch + 1, num_epochs, i + 1, loss_train.item(), loss_val.item(), accu,
        _lr_scheduler.get_last_lr()[0]))

def train(num_epochs, _model, _device, _train_loader, _val_data, _optimizer, _lr_scheduler, criterion):
    _model.train()  # 设置模型为训练模式
    val_x, val_y = _val_data[0].to(_device), _val_data[1].to(_device)
    for epoch in range(num_epochs):
        # 从迭代器抽取图片和标签
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            output = _model(samples)
            # 计算损失函数值
            loss_train = criterion(output, labels)
            # 损失值后向传播
            loss_train.backward()
            # 更新模型参数
            _optimizer.step()
            # 优化器内部参数梯度必须变为0
            _optimizer.zero_grad()
                 
            if (i + 1) % 50 == 0:
                check_loss(_model, val_x, val_y, criterion, epoch, num_epochs, i, loss_train, _lr_scheduler)
        _lr_scheduler.step()      


def test(ConvModel, test_xy, _model, _device, criterion):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
 
    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        x, label = test_xy[0].to(_device), test_xy[1].to(_device)
        output = ConvModel(x)
        loss += criterion(output, label).item()  # 添加损失值
        pred = output.argmax(axis=1)  # 找到概率最大的下标，为输出值
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来
        result = (pred == label)
        wrong = torch.where(result == False)[0]
        print(wrong)
 
    print('test avg loss: {:.4f}, accu: {}/{} ({:.3f}%)'.format(
        loss, correct, x.shape[0],
        100. * correct / x.shape[0]))

def load_shape_data(train_file_name, test_file_name, mode="image"):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data(mode=mode)
    #data_loader.to_onehot(data_loader.num_classes)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.9)
    return data_loader

def main():
    # 预设网络超参数 （所谓超参数就是可以人为设定的参数   
    BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
    EPOCHS = 50  # 总共训练迭代的次数
    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    learning_rate = 0.01  # 设定初始的学习
    # 加载训练集
    data_loader = load_shape_data("train_shape_4.npz", "test_shape_4.npz", mode="image")
    train_x, train_y = data_loader.get_train()
    val_x, val_y = data_loader.get_val()
    test_x, test_y = data_loader.get_test()
    test_x = data_loader.StandardScaler_pred_X(test_x)

    torch_train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y.squeeze()))
    torch_test_x, torch_test_y = torch.FloatTensor(test_x), torch.LongTensor(test_y.squeeze())
    torch_val_x, torch_val_y = torch.FloatTensor(val_x), torch.LongTensor(val_y.squeeze())

    train_loader = DataLoader(
        dataset=torch_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )   
    
    # 初始化模型 将网络操作移动到GPU或者CPU
    ConvModel = ConvNet().to(DEVICE)

    print(ConvModel)

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # 定义模型优化器：输入模型参数，定义初始学习率
    optimizer = torch.optim.Adam(ConvModel.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(ConvModel.parameters(), lr=learning_rate)
    # 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    train(EPOCHS, ConvModel, DEVICE, train_loader, (torch_val_x, torch_val_y), optimizer, exp_lr_scheduler, criterion)
    torch.save(ConvModel.state_dict(), "D:/Gitee/nnbasic/深度学习基础/第14章 二维卷积网络/src/model/ConvModel.pth")
    test(ConvModel, (torch_test_x, torch_test_y), ConvModel, DEVICE, criterion)


def parse_pth():
    file = "D:/Gitee/nnbasic/深度学习基础/第14章 二维卷积网络/src/model/ConvModel.pth"    
    params = torch.load(file)
    print(params.keys())
    for k,v in params.items():
        print(k)
        if k == "classifier.0.weight":
            print(v.shape)
            with open("D:/Gitee/nnbasic/深度学习基础/第14章 二维卷积网络/src/model/ConvModel_" +k+".txt", "w") as f:
                for i in range(v.shape[1]):
                    for j in range(v.shape[0]):
                        f.write(str(v[j,i].item()))
                        f.write(" ")
                    f.write("\n")
        else:
            v1 = v.reshape(-1,1)
            # 建立一个文本文件用于保存 v1 中的数值
            with open("D:/Gitee/nnbasic/深度学习基础/第14章 二维卷积网络/src/model/ConvModel_" +k+".txt", "w") as f:
                for i in range(v1.shape[0]):
                    f.write(str(v1[i].item()) + "\n")
        

if __name__=="__main__":
    start = time.time()    
    #main()
    end = time.time()
    print("time:", end-start)
    parse_pth()


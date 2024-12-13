import os
import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim  # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import time
from common.DataLoader_14 import DataLoader_14
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)


# 设计模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            # 28 -> 30 -> 28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 28 -> 28
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # (28-2)/2+1 = 14
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14 -> 16 -> 14
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 14 -> 16 -> 14
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 14 -> 14
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # 7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1176, 256),  # 864
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10), 
        )
    # 前向传递函数    
    def forward(self, x):
        # 经过特征提取层
        x = self.features(x)
        # 输出结果必须展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test(_test_loader, _model, _device, loss_func):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        count = 0
        for _data, _target in _test_loader:
            count += 1
            x, y = _data.to(_device), _target.to(_device)
            output = _model(x.reshape(-1, 1, 28, 28))
            loss += loss_func(output, y).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
 
    loss /= count
    print('测试集: \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))
    return correct

def augment_data_loader(BATCH_SIZE):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_filename = os.path.join(current_dir, "data", "FashionMNIST_Train1.npz")
    test_filename = os.path.join(current_dir, "data", "FashionMNIST_Test1.npz")
    data_loader = DataLoader_14(train_filename, test_filename)
    data_loader.load_npz_data()
    data_loader.StandardScaler_X(is_image=True)
    train_x, train_y = data_loader.get_train()
    test_x, test_y = data_loader.get_test()
    test_x = data_loader.StandardScaler_pred_X(test_x)

    torch_train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y.squeeze()))
    torch_test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y.squeeze()))

    train_loader = DataLoader(
        dataset=torch_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )   

    test_loader = DataLoader(
        dataset=torch_test_dataset,
        batch_size=BATCH_SIZE,
    )   
    return train_loader, test_loader

def find_wrong_class_n(y, wrong_idx, class_id, count):
    pos = []
    for i in wrong_idx:
        if y[i] == class_id:
            pos.append(i)
            if len(pos) == count:
                return pos
            
def eval_model():
    DEVICE = torch.device("cpu")
    ConvModel = ConvNet().to(DEVICE)
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", "ConvModel_AlexNet_Aug1.pth")
    ConvModel.load_state_dict(torch.load(train_pth))
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    train_loader, test_loader = augment_data_loader(10000)
    test(test_loader, ConvModel, DEVICE, loss_func)

    wrong_index = []
    for _data, _target in test_loader:
        x, y = _data.to(DEVICE), _target.to(DEVICE)
        output = ConvModel(x)
        y_pred = output.argmax(axis=1)
        result = (y_pred == y)
        wrong = torch.where(result == False)[0]
        wrong_index.append(wrong)
    wrong_index = torch.concat(wrong_index)
    print(wrong_index.shape)

    names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

    # 看前 10 个判别错的测试集样本
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8,4))
    for i in range(10):
        poss = find_wrong_class_n(y, wrong_index, i, 2)
        pos = poss[0]
        ax = axes[i//5,i%5]
        label_id = y[pos]
        predict_id = y_pred[pos]
        img = x[pos].reshape(28, 28)
        ax.imshow(img, cmap="gray_r")
        ax.set_title(names[label_id] + "(" + names[predict_id] + ")")
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    # 混淆矩阵
    # 手工计算混淆矩阵值
    confusion_matrix = torch.zeros((10, 10))
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y[i]:  # 预测正确
            confusion_matrix[y[i], y[i]] += 1 # 对角线，True Positive
        else:  # 预测错误
            confusion_matrix[y[i], y_pred[i]] += 1 # FN,FP,TN
    print(confusion_matrix)
    plt.imshow(confusion_matrix, cmap='autumn_r')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, "%d"%(confusion_matrix[i, j]), ha='center', va='center')
    tick = range(10)
    plt.xticks(tick, names)
    plt.yticks(tick, names)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    plt.show()


if __name__=="__main__":
    eval_model()    

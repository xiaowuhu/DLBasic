import os
import torch
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def create_data_loader(batch_size):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
        batch_size=batch_size)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
        ])),
        batch_size=batch_size)
    
    return train_loader, test_loader

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def find_class_img(labels, id, n):
    poss = []
    for i in range(len(labels)):
        if labels[i] == id:
            poss.append(i)
    return poss

def read_one_file():
    p = os.path.join(os.getcwd(), "data/ch15/cifar-10/data_batch_1")
    d = unpickle(p)

    print(d.keys())
    print(d[b'batch_label'])
    print(d[b'labels'])

    names = ["飞机","汽车","鸟","猫","鹿","狗","蛙","马","船","卡车"]

    fig, axes = plt.subplots(nrows=10, ncols=11, figsize=(6,6)) 
    img = d[b'data']
    for i in range(10):
        poss = find_class_img(d[b'labels'], i, 10)
        ax = axes[i, 0]
        ax.text(0,0,names[i])
        ax.axis("off")

        for j in range(10):
            ax = axes[i, j+1]
            image = img[poss[j]]
            red_image = image[:1024].reshape(32,32)
            green_image = image[1024:2048].reshape(32,32)
            blue_image = image[2048:].reshape(32,32)
            result_img = np.ones((32, 32, 3), dtype=np.uint8)
            result_img[:,:,0] = red_image
            result_img[:,:,1] = green_image
            result_img[:,:,2] = blue_image
            ax.imshow(result_img)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

    plt.show()

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
    read_one_file()
    batch_size = 1000
    train_loader, test_loader = create_data_loader(batch_size)
    mean, std = compute_mean_std_batch(train_loader)
    print("训练集均值和标准差:\n", mean, std)
    # compute_mean_std_whole(train_loader)
    print("computing...")
    mean, std = compute_mean_std_batch(test_loader)
    print("测试集均值和标准差:\n", mean, std)

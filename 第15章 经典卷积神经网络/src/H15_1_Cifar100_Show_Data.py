import torch
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=9)

def create_data_loader(batch_size):
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
        batch_size=batch_size)  # 指明批量大小，打乱，这是后续训练的需要。
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
        ])),
        batch_size=batch_size)
    
    return train_loader, test_loader

def find_class_img(labels, id, n):
    poss = []
    for i in range(len(labels)):
        if labels[i] == id:
            poss.append(i)
    return poss

def show_class_image(data_loader):
    fig, axes = plt.subplots(nrows=5, ncols=20, figsize=(8,6)) 
    for x, y in data_loader:
        for id in range(100):
            poss = find_class_img(y, id, 1)
            pos = poss[0]
            img = x[pos].numpy().transpose(1,2,0)
            #ax = axes[id%5, id//5]
            ax = axes[id//20, id%20]
            ax.set_xlabel(str(id))
            #ax.axis("off")
            ax.imshow(img)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        
        plt.show()
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(6,6)) 
        people = [2,11,35,46,98]
        names = ["婴儿", "男孩", "女孩", "男人", "女人"]
        for i in range(5):
            ax = axes[i, 0]
            ax.text(0.2, 0.2, names[i])
            ax.axis("off")            
            id = people[i]
            poss = find_class_img(y, id, 5)
            for j in range(5):
                ax = axes[i, j+1]
                img = x[poss[j]].numpy().transpose(1,2,0)
                ax.imshow(img)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
        plt.show()
        break
    
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
    train_loader, test_loader = create_data_loader(batch_size)
    show_class_image(train_loader)
    
    mean, std = compute_mean_std_batch(train_loader)
    print("训练集均值和标准差:\n", mean, std)
    # compute_mean_std_whole(train_loader)
    print("computing...")
    mean, std = compute_mean_std_batch(test_loader)
    print("测试集均值和标准差:\n", mean, std)

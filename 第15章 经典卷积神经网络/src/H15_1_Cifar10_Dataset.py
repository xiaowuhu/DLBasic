import torch
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.utils
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
import numpy as np

def raw_data_loader(BATCH_SIZE):
    org_trainset = datasets.CIFAR10("data", train=True, download=True)
    choice = list(range(len(org_trainset))) # 50000
    np.random.shuffle(choice)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # 数据增强，在训练后期使用
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomAffine(0, translate=(0.1,0.1)),
        # transforms.RandomCrop(32, padding=4),
    ])    
    new_trainset = my_ds(org_trainset, choice[:45000], transform)
    validation_set = my_ds(org_trainset, choice[45000:], transform)

    # 加载训练集
    train_loader = torch.utils.data.DataLoader(new_trainset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader

class my_ds(torch.utils.data.Dataset):
    def __init__(self, trainset, choice, transform):
        self.transform = transform
        self.images = trainset.data[choice].copy()
        self.labels = [trainset.targets[i] for i in choice]

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        sample = (image, label)
        return sample


if __name__=="__main__":
    train_loader, val_loader, test_loader = raw_data_loader(64)
    print("数据集\t总数\t批次\t批量")
    print("训练集\t{}\t{}\t64".format(len(train_loader.dataset), len(train_loader)))
    print("验证集\t{}\t{}\t64".format(len(val_loader.dataset), len(val_loader)))
    print("测试集\t{}\t{}\t64".format(len(test_loader.dataset), len(test_loader)))

import matplotlib.pyplot as plt
import os
from torchvision.transforms import v2 as transforms
#from torchvision import transforms

plt.rc('font', size=9)


def Rotation(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(-30,-10)),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def Affine(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.2)),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def CenterCrop(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((int(source_img.shape[0]*0.6), int(source_img.shape[1]*0.6))),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def ResizeCrop(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((source_img.shape[0], source_img.shape[1]), scale=(0.8,0.8)),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def Perspective(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomPerspective(distortion_scale=0.4, p=1),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def Elastic(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.ElasticTransform(alpha=100.0),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def HorizontalFlip(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C

def VerticalFlip(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=1),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C


def GaussianBlur(source_img):
    # 输入的形状是(H,W,C)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ])
    img_new = trans(source_img)  # C,H,W
    return img_new.numpy().transpose(1,2,0) # H,W,C


if __name__=="__main__":

    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    img_path = os.path.join(current_dir, "data", "sample.png")
    img = plt.imread(img_path)[:, :, 0:3]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,6)) 
    axes[0,0].imshow(Rotation(img))
    axes[0,0].set_title("Rotation")
    axes[0,1].imshow(ResizeCrop(img))
    axes[0,1].set_title("ResizeCrop")
    axes[0,2].imshow(CenterCrop(img))
    axes[0,2].set_title("CenterCrop")

    axes[1,0].imshow(Affine(img))
    axes[1,0].set_title("Affine")
    axes[1,1].imshow(img)
    axes[1,1].set_title("Source")
    axes[1,2].imshow(Perspective(img))
    axes[1,2].set_title("Perspective")

    axes[2,0].imshow(Elastic(img))
    axes[2,0].set_title("Elastic")
    axes[2,1].imshow(HorizontalFlip(img))
    axes[2,1].set_title("HorizontalFlip")
    axes[2,2].imshow(VerticalFlip(img))
    axes[2,2].set_title("VerticalFlip")

    plt.show()

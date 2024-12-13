import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        # 1 x 1
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # 1 x 1 -> 3 x 3
        self.c2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 1 x 1 -> 5 x 5
        self.c3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )        
        # P 3 x 3 -> C 1 x 1
        self.c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )        
    
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        cat = torch.cat((x1, x2, x3, x4), dim=1)
        return cat, x1.detach().numpy(), x2.detach().numpy(), x3.detach().numpy(), x4.detach().numpy()

def create_gray_image():
    circle_pic = os.path.join(os.path.dirname(__file__), "data", "circle.png")
    img = mpimg.imread(circle_pic)
    a = np.array([0.299, 0.587, 0.114, 0])
    gray = np.dot(img, a)
    return gray.astype(np.float32)

if __name__=="__main__":
    gray_img = create_gray_image()
    print(gray_img.shape)
    batch_img = np.expand_dims(gray_img, 0)
    print(batch_img.shape)
    x = torch.from_numpy(batch_img)
    x.requires_grad_(requires_grad=False)
    model = Inception()
    output, x1, x2, x3, x4 = model(x)
    print(output.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8,3)) 
    axes[0].imshow(batch_img.transpose(1,2,0), cmap="gray_r")
    axes[0].set_title("source")
    axes[1].imshow(x1.transpose(1,2,0), cmap="gray_r")
    axes[1].set_title("1x1")
    axes[2].imshow(x2.transpose(1,2,0), cmap="gray_r")
    axes[2].set_title("3x3")
    axes[3].imshow(x3.transpose(1,2,0), cmap="gray_r")
    axes[3].set_title("5x5")
    axes[4].imshow(x4.transpose(1,2,0), cmap="gray_r")
    axes[4].set_title("maxpool")
    plt.show()

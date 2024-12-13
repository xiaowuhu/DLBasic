import os
import torch
import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数

def print_training_progress(epoch, num_epochs, step, total_step, step_loss, lr_scheduler):
    # todo: add validation set here
    rate = (step + 1) / total_step
    prefix = "*" * int(rate * 50)
    suffix = "." * int((1-rate) * 50)
    print("\rEpoch:{}/{} (lr={:.5f}) {:^4.0f}%[{}->{}]{:.4f}".format(
        epoch + 1, num_epochs, lr_scheduler.get_last_lr()[0],
        int(rate * 100), prefix, suffix, step_loss),
        end="")

def print_test_progress(running_loss, total_count, test_loss, correct):
    print('Running Loss:{:.4f}, Test Loss: {:.4f}, Accu: {}/{} ({:.2f}%)\n'.format(
        running_loss, test_loss, correct, total_count, 100. * correct / total_count))


def train_model(num_epochs, model, device, train_loader, test_loader, optimizer, lr_scheduler, loss_func, name, best_correct=0):
    best_correct = best_correct  # 自定义初始 best 便于及时保存结果，否则从 0 开始自动增长
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
            # 显示进度条和当前批次的误差
            print_training_progress(epoch, num_epochs, step, len(train_loader), step_loss, lr_scheduler)
        lr_scheduler.step()
        print()
        test_loss, correct = test_model(test_loader, model, device, loss_func)
        print_test_progress(running_loss/len(train_loader), len(test_loader.dataset), test_loss, correct)
        if correct > best_correct:
            save_model(model, name)
            best_correct = correct

def test_model(test_loader, model, device, loss_func):
    model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        for test_x, test_y in test_loader:
            x, y = test_x.to(device), test_y.to(device)
            predict = model(x)
            loss += loss_func(predict, y) # 添加损失值
            pred = predict.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
 
    loss /= len(test_loader)
    return loss, correct

def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

# def data_loader(BATCH_SIZE):
#     # 加载训练集
#     train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR100('data', train=True, download=True,
#             transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.51, 0.48, 0.44), (0.27, 0.25, 0.28)),
#             ])),
#         batch_size=BATCH_SIZE, shuffle=True)  # 指明批量大小，打乱，这是后续训练的需要。
    
#     test_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR100('data', train=False, download=True,
#             transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.51, 0.48, 0.44), (0.27, 0.25, 0.28)),
#         ])),
#         batch_size=BATCH_SIZE)
#     return train_loader, test_loader

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    model.load_state_dict(torch.load(model_pth, map_location=device))


def eval_model(net:nn.Module, test_loader, name:str):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model:nn.Module = net.to(DEVICE)
    load_model(model, name, DEVICE)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    loss, correct = test_model(test_loader, model, DEVICE, loss_func)
    print("Test Loss: {:.4f}, Correct: {}/{} ({:.2f}%)".format(loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

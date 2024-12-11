import os
from common.NeuralNet_1 import NeuralNet_1
from common.DataLoader_1 import DataLoader_1
from H1_1_ShowData import show_data

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train1.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_1(file_path)
    data_loader.load_data()
    print("训练神经网络...")
    lr = 0.000001
    w = 0
    epoch = 1000
    nn = NeuralNet_1(data_loader, w, lr)
    nn.train(epoch)
    print("权重值 w =", nn.w)
    print("预测...")
    area = 100
    price = nn.predict(area)
    print("%f平米的房屋价格为%f万元" %(area, price))

    show_data(nn.w)

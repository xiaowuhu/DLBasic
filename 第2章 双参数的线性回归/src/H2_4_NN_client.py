import os
from common.NeuralNet_2 import NeuralNet_2
from common.DataLoader_2 import DataLoader_2
from H2_1_ShowData import show_data

if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()
    print("训练神经网络...")
    lr = 0.00001
    w = 0
    b = 0
    epoch = 100000
    nn = NeuralNet_2(data_loader, w, b, lr)
    nn.train(epoch)
    print("权重值 w = %f, 偏移值 b = %f" %(nn.w, nn.b))
    print("预测...")
    area = 120
    price = nn.predict(area)
    print("%f平米的房屋价格为%f万元" %(area, price))

    show_data(nn.w, nn.b)

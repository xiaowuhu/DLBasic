import os
from common.DataLoader_2 import DataLoader_2
from H2_3_LeastSquare import calculate_w_b


if __name__ == '__main__':
    # 准备数据
    print("加载数据...")
    file_name = "train2.txt"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    data_loader = DataLoader_2(file_path)
    data_loader.load_data()
    data_loader.normalize_train_data()
    X,Y = data_loader.train_x, data_loader.train_y
    w, b = calculate_w_b(X, Y)
    print("w = %f, b = %f" % (w, b))

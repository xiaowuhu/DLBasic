import os
import numpy as np
from common.DataLoader_14 import DataLoader_14
from common.Module import Sequential
import matplotlib.pyplot as plt

from H14_7_Shape_2_CNN_Train import build_cnn_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def load_shape_data(train_file_name, test_file_name, mode="image"):
    train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", train_file_name)
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", test_file_name)
    data_loader = DataLoader_14(train_file_path, test_file_path)
    data_loader.load_npz_data(mode=mode)
    data_loader.to_onehot(2)
    data_loader.StandardScaler_X(is_image=True)
    data_loader.shuffle_data()
    data_loader.split_data(0.9)
    return data_loader


def test_model(data_loader: DataLoader_14, model: Sequential, model_name):
    model.load(model_name)
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))

    predict = model.forward(x)
    predict = np.argmax(predict, axis=1)
    label = np.argmax(test_label, axis=1)
    result = (predict == label)
    wrong_idxes = np.where(result == False)[0]
    # 看前 n 个
    n = int(data_loader.num_test * (1-test_accu) + 0.1)
    n = min(n, 10)
    fig = plt.figure(figsize=(8, 3))

    for i in range(n):
        ax = fig.add_subplot(2, 5, i+1)
        id = wrong_idxes[i]
        label_id = label[id]
        predict_id = predict[id]
        img = test_x[id].reshape(28, 28)
        ax.imshow(img, cmap="gray_r")
        ax.set_title("%d(%d)"%(label_id, predict_id))
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()



def min_max_normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def show_img(ax, img, norm=True):
    if norm == True:
        img = min_max_normalize(img)
    im = ax.imshow(img, cmap='binary')
    return im

def show_result_kernel(data_loader:DataLoader_14, model: Sequential):
    # 可视化卷积核
    kernel = model.operator_seq[0].WB.W
    kernel_count = kernel.shape[0]
    fig, axes = plt.subplots(nrows=kernel_count*2, ncols=6, figsize=(6,3))

    kernel_names = ["kernel-1", "kernel-2"]
    for i in range(kernel_count * data_loader.num_classes):
        ax = axes[i, 0]
        show_img(ax, kernel[i//2,0], norm=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(kernel_names[i//2])
        

    test_x, test_y = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    conv_z = model.operator_seq[0].forward(x)  # conv
    relu_z = model.operator_seq[1].forward(conv_z)  # relu
    pool_z = model.operator_seq[2].forward(relu_z)  # pool

    idx_rect_sample = 10
    start_offset = 200
    class_names = ["circle", "rectangle"]
    for i in range(kernel_count):
        for j in range(data_loader.num_classes):
            id = j*start_offset+idx_rect_sample
            # 原始图片
            ax = axes[i*kernel_count+j, 1]
            show_img(ax, x[id, 0], norm=True)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(class_names[j])
            # conv 结果
            ax = axes[i*kernel_count+j, 2]
            im = show_img(ax, conv_z[id, i], norm=True)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i * kernel_count + j == 0:
                ax.set_title("conv")
            # relu 结果
            ax = axes[i*kernel_count+j, 3]
            show_img(ax, relu_z[id, i], norm=True)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i * kernel_count + j == 0:
                ax.set_title("relu")
            # pool 结果
            ax = axes[i*kernel_count+j, 4]
            show_img(ax, pool_z[id, i], norm=True)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i * kernel_count + j == 0:
                ax.set_title("pool")

    for i in range(kernel_count * data_loader.num_classes):
        axes[i,5].axis("off")
    plt.colorbar(im, ax=axes[:,5].ravel().tolist())
    plt.show()

def show_result_size_pos(data_loader:DataLoader_14, model: Sequential):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))

    test_x, test_y = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    conv_z = model.operator_seq[0].forward(x)  # conv
    relu_z = model.operator_seq[1].forward(conv_z)  # relu
    pool_z = model.operator_seq[2].forward(relu_z)  # pool

    for i in range(4):
        for j in range(4):
            # 原始图片 圆形
            ax = axes[i, 0]
            show_img(ax, x[i, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("circle")
            # conv 结果
            ax = axes[i, 1]
            im = show_img(ax, conv_z[i, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("conv")
            # relu 结果
            ax = axes[i, 2]
            show_img(ax, relu_z[i, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("relu")
            # pool 结果
            ax = axes[i, 3]
            show_img(ax, pool_z[i, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("pool")

            # 原始图片 方形
            ax = axes[i, 4]
            show_img(ax, x[i+200, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("rectangle")
            # conv 结果
            ax = axes[i, 5]
            im = show_img(ax, conv_z[i+200, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("conv")
            # relu 结果
            ax = axes[i, 6]
            show_img(ax, relu_z[i+200, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("relu")
            # pool 结果
            ax = axes[i, 7]
            show_img(ax, pool_z[i+200, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("pool")

    plt.show()



if __name__=="__main__":
    model_name = "Shape_2_conv_14_6_CNN_1_pad1" # 一个卷积核+padding
    model = build_cnn_model("1+padding")

    # model_name = "Shape_2_conv_14_6_CNN_2"    # 两个卷积核
    #model = build_cnn_model("2")

    # model_name = "Shape_2_conv_14_6_CNN_1"    # 一个卷积核
    #model = build_cnn_model("1")
       
    data_loader = load_shape_data("train_shape_2.npz", "test_shape_2.npz", mode="image")
    test_model(data_loader, model, model_name)
    show_result_kernel(data_loader, model)
    show_result_size_pos(data_loader, model)

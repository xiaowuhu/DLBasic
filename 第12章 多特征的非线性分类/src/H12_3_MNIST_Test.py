import numpy as np

from common.DataLoader_12 import DataLoader_12
from common.Module import Sequential
import matplotlib.pyplot as plt

from H12_3_MNIST_Train import build_model

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)


def load_minist_data():
    file_path = "../data/MNIST/"
    data_loader = DataLoader_12(file_path)
    data_loader.load_MNIST_data("vector")
    data_loader.to_onehot(10)
    data_loader.StandardScaler_X(is_image=True)
    return data_loader

def test_model(data_loader: DataLoader_12, model: Sequential):
    test_x, test_label = data_loader.get_test()
    x = data_loader.StandardScaler_pred_X(test_x)
    test_loss, test_accu = model.compute_loss_accuracy(x, test_label)
    print("测试集: loss %.4f, accu %.4f" %(test_loss, test_accu))

    predict = model.forward(x)
    predict = np.argmax(predict, axis=1)
    label = np.argmax(test_label, axis=1)
    result = (predict == label)
    wrong_idxes = np.where(result == False)[0]
    # 看前 10 个
    fig = plt.figure(figsize=(8, 3))

    for i in range(10):
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


if __name__=="__main__":
    model = build_model()
    data_loader = load_minist_data()
    model.load("model_12_3_SGD")
    test_model(data_loader, model)

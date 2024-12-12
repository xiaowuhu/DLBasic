import numpy as np
import common.Layers as layer
import common.Activators as activator
import common.LossFunctions as loss
from common.Module import Module, Sequential
from H10_5_GradientCheck import *

def check():
    # 建立字典，生成矢量数据
    dict_param = {
        "W1": model.operator_seq[0].WB.W, "B1": model.operator_seq[0].WB.B, 
        "W2": model.operator_seq[2].WB.W, "B2": model.operator_seq[2].WB.B, 
    }
    dict_grads = {
        "dW1": model.operator_seq[0].WB.dW, "dB1": model.operator_seq[0].WB.dB, 
        "dW2": model.operator_seq[2].WB.dW, "dB2": model.operator_seq[2].WB.dB, 
    }

    f_x_vector, param_shape_dict = parameters_to_vector(dict_param)
    automatic_diff = gradients_to_vector(dict_grads)

    num_x = f_x_vector.shape[0]
    numerical_diff = np.zeros((num_x, 1))
    delta = 1e-5

    # for each of the all parameters in w,b array
    for x in range(num_x):
        # 右侧
        f = np.copy(f_x_vector)  # 每次都使用原始的数据
        f[x][0] = f_x_vector[x][0] + delta
        dict_parameters = vector_to_parameters(f, param_shape_dict)
        f_right = calculate_loss(model, dict_parameters, X, Y)
        # 左侧
        f = np.copy(f_x_vector)
        f[x][0] = f_x_vector[x][0] - delta
        dict_parameters = vector_to_parameters(f, param_shape_dict)
        f_left = calculate_loss(model, dict_parameters, X, Y)
        # 式（5.8.7)
        numerical_diff[x] = (f_right - f_left) / (2 * delta)
    # end for
    numerator = np.linalg.norm(automatic_diff - numerical_diff)  # np.linalg.norm 二范数
    denominator = np.linalg.norm(numerical_diff) + np.linalg.norm(automatic_diff)
    difference = numerator / denominator
    return difference


if __name__=="__main__":
    # 初始化网络参数
    model = Sequential(
        layer.Linear(1, 2),
        activator.my_activator(),     # 激活层
        layer.Linear(2, 1),   # 线性层2（输出层）
    )
    model.set_classifier_function(layer.my_binary_classifier())
    model.set_loss_function(loss.my_loss())    # 生成随机测试样本
    num_sample = 8
    X = np.random.rand(num_sample, 1)
    Y = np.random.binomial(1, 0.5, size=(num_sample,1))

    print("计算自动微分梯度...")
    # 进行正向计算和反向传播各一次
    A = model.forward(X)
    model.backward(A, Y)
    # 自动微分法的梯度准备好了
    
    difference = check()
    print('diference =', difference)
    if difference<1e-7:
        print("完全没有问题! 喝一杯红酒祝一下!")
    elif difference<1e-4:
        print("虽然有一些误差但是还可以接受，喝一杯啤酒去吧。")
    elif difference<1e-2:
        print("误差比较高，先喝一杯白酒提提神，然后需要检查代码是否有问题。")
    else:
        print("误差太大，啥都别喝了，赶紧改代码!")
    
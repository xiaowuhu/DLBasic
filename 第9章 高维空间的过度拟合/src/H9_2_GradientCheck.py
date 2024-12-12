import numpy as np
from H9_3_NN_over_fitting import NN

# 把所有参数拼接成一个向量
def parameters_to_vector(dict_params):
    WB_values_vector = None
    WB_shapes_dict = {}
    for key in dict_params.keys():
        WB_shapes_dict[key] = dict_params[key].shape
        # flatten parameter
        new_vector = np.reshape(dict_params[key], (-1,1)) # N行1列
        if WB_values_vector is None:
            WB_values_vector = new_vector
        else:
            WB_values_vector = np.concatenate((WB_values_vector, new_vector), axis=0)
 
    return WB_values_vector, WB_shapes_dict

# 把所有梯度拼接成一个向量
def gradients_to_vector(dict_grads):
    dWB_values = None
    for key in dict_grads.keys():
        # flatten parameter
        new_vector = np.reshape(dict_grads[key], (-1,1))
        if dWB_values is None:
            dWB_values = new_vector
        else:
            dWB_values = np.concatenate((dWB_values, new_vector), axis=0)
 
    return dWB_values

# 把所有参数还原成原始形状
def vector_to_parameters(J_xi, x_shape):
    dict_params = {}
    start = 0
    for wb_name, wb_shape in x_shape.items():
        end = start + np.prod(wb_shape)
        dict_params[wb_name] = J_xi[start:end].reshape(wb_shape)
        start = end
    return dict_params

# 计算损失函数值
def calculate_loss(model, dict_Param, X, Y):
    model.linear1.W = dict_Param["W1"]
    model.linear1.B = dict_Param["B1"]
    model.linear2.W = dict_Param["W2"]
    model.linear2.B = dict_Param["B2"]
    model.linear3.W = dict_Param["W3"]
    model.linear3.B = dict_Param["B3"]
    model.linear4.W = dict_Param["W4"]
    model.linear4.B = dict_Param["B4"]
    Z = model.forward(X)
    loss = model.loss.forward(Z, Y)
    return loss


if __name__=="__main__":
    # 初始化网络参数
    model = NN()
    print("权重矩阵初始化...")

    # 生成随机测试样本
    num_sample = 8
    X = np.random.rand(num_sample, 1)
    Y = np.random.rand(num_sample, 1)

    print("计算自动微分梯度...")
    # 进行正向计算和反向传播各一次
    Z = model.forward(X)
    model.backward(Z, Y)
    # 自动微分法的梯度准备好了
    
    # 建立字典，生成矢量数据
    dict_param = {
        "W1": model.linear1.W, "B1": model.linear1.B, 
        "W2": model.linear2.W, "B2": model.linear2.B,
        "W3": model.linear3.W, "B3": model.linear3.B,
        "W4": model.linear4.W, "B4": model.linear4.B,
    }
    dict_grads = {
        "dW1": model.linear1.dW, "dB1": model.linear1.dB,
        "dW2": model.linear2.dW, "dB2": model.linear2.dB,
        "dW3": model.linear3.dW, "dB3": model.linear3.dB,
        "dW4": model.linear4.dW, "dB4": model.linear4.dB,
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
    print('diference =', difference)
    if difference<1e-7:
        print("完全没有问题! 喝一杯红酒祝一下!")
    elif difference<1e-4:
        print("虽然有一些误差但是还可以接受，喝一杯啤酒去吧。")
    elif difference<1e-2:
        print("误差比较高，先喝一杯白酒提提神，然后需要检查代码是否有问题。")
    else:
        print("误差太大，啥都别喝了，赶紧改代码!")
    
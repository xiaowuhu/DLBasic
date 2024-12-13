import numpy as np
import common.Layers as layer

from common.Module import Module, Sequential


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
    model.operator_seq[0].WB.W = dict_Param["W1"]
    #model.operator_seq[0].WB.B = dict_Param["B1"]
    Z = model.forward(X)
    loss = model.compute_loss(Z, Y)
    return loss


def conv_model():
    model = Sequential(
        layer.Conv1d((1,5), (1,3), stride=1),
        layer.Flatten((1,3),3)
    )
    model.set_classifier_loss_function(layer.SoftmaxCrossEntropy())
    return model


if __name__=="__main__":
    # 初始化网络参数
    model = conv_model()
    # 生成随机测试样本
    num_sample = 8
    X = np.random.rand(num_sample, 5).reshape(num_sample, 1, 5)
    Y = np.random.rand(num_sample, 3)
    #X = np.array([[1.0,2,3,4,5.0]])

    print("计算自动微分梯度...")
    # 进行正向计算和反向传播各一次
    A = model.forward(X)
    Y = np.reshape(Y, A.shape)

    model.backward(A, Y)
    # 自动微分法的梯度准备好了
    
    # 建立字典，生成矢量数据
    dict_param = {
        "W1": model.operator_seq[0].WB.W,
        #"W2": model.operator_seq[0].WB.W,
        #"W3": model.operator_seq[0].WB.W
    }
    dict_grads = {
        "dW1": model.operator_seq[0].WB.dW,
        #"dW2": model.operator_seq[0].WB.dW,
        #"dW3": model.operator_seq[0].WB.dW,
    }

    f_x_vector, param_shape_dict = parameters_to_vector(dict_param)
    automatic_diff = gradients_to_vector(dict_grads)
    print("自动微分结果：", automatic_diff)

    num_x = f_x_vector.shape[0]
    numerical_diff = np.zeros((num_x, 1))
    delta = 1e-5

    # for each of the all parameters in w,b array
    for x in range(num_x):
        # 右侧
        f = np.copy(f_x_vector)  # 每次都使用原始的数据
        f[x][0] = f_x_vector[x][0] + delta
        # f[x+2][0] = f_x_vector[x][0] + delta
        # f[x+4][0] = f_x_vector[x][0] + delta
        dict_parameters = vector_to_parameters(f, param_shape_dict)
        f_right = calculate_loss(model, dict_parameters, X, Y)
        # 左侧
        f = np.copy(f_x_vector)
        f[x][0] = f_x_vector[x][0] - delta
        # f[x+2][0] = f_x_vector[x][0] - delta
        # f[x+4][0] = f_x_vector[x][0] - delta
        dict_parameters = vector_to_parameters(f, param_shape_dict)
        f_left = calculate_loss(model, dict_parameters, X, Y)
        # 式（5.8.7)
        numerical_diff[x] = (f_right - f_left) / (2 * delta)
    # end for
    print("数值微分结果:", numerical_diff)
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
    
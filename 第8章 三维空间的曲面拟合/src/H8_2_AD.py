import numpy as np
import common.Layers as layer
from common.Module import Module
from common import BatchNorm1d

# 变量节点
class variable_node(object):
    def __init__(self, name: str, value, is_constant: bool=False) -> None:
        self.name = name
        self.value = value
        self.grad = np.zeros_like(value)
        self.is_constant = is_constant

    def set_value(self, value) -> None:
        assert(self.value.shape == value.shape)
        self.value = value

    def set_grad(self, value) -> None:
        #assert(self.grad.shape == value.shape) # 有些情况下不相等
        # assume already set to zero grad before
        # so need accumulate the grad value
        self.grad += value  

    def __str__(self) -> str:
        return self.name
    

# 计算节点
class compute_node(object):
    def __init__(self, 
        op: layer.Operator, 
        name: str, 
        input: list[variable_node],
        output: variable_node
    ) -> None:
        self.name: str = name
        self.op: layer.Operator = op
        self.input_vars: list[variable_node] = input
        self.output_var: variable_node = output  # assume only one output

    def get_input_var_values(self):
        var_values = []
        for input_var in self.input_vars:
            var_values.append(input_var.value)
        return var_values

    def get_input_vars(self) -> list[variable_node]:
        return self.input_vars

    # assume only one output
    def get_output_var(self) -> variable_node:
        return self.output_var

    def __str__(self) -> str:
        info = self.name + "("
        for input_var in self.input_vars:
            info += input_var.name + ","
        info += ") -> " + self.output_var.name
        return info

# 计算图
class compute_graph(object):
    def __init__(self) -> None:
        self.list_operator: list[compute_node] = []
        self.list_variable: list[variable_node] = []
        self.dict_name2var = {}
        self.dict_output_var_name_to_op_node = {}
        
    def add_variable(self, value, name: str, is_constant:bool = False) -> variable_node:
        variable = variable_node(name, value, is_constant=is_constant)
        self.list_variable.append(variable)
        self.dict_name2var[name] = variable
        return variable

    def add_operator(self, op_node: compute_node) -> None:
        self.list_operator.append(op_node)
        var = op_node.get_output_var()
        self.dict_output_var_name_to_op_node[var.name] = op_node

    # 根据 var 的名字找输出此 var 的 op
    def get_op_by_output_name(self, var_name: str) -> compute_node:
        op_node = self.dict_output_var_name_to_op_node.get(var_name, None)
        return op_node

    def get_var_by_name(self, var_name: str) -> variable_node:
        var = self.dict_name2var.get(var_name, None)
        return var

    # 把所有 var 的 grad 清零
    def zero_grads(self):
        for var in self.list_variable:
            var.grad.fill(0)

    # 当所有的 Op 和 var 都添加完毕后，调用此方法
    # x is input
    def build_forward_graph(self, x):
        print(" ---- 开始正向计算 ----")
        # 初始化 graph input
        self.list_variable[0].set_value(x)
        # assume 在list中的计算顺序是正确的
        for op in self.list_operator:
            var_values = op.get_input_var_values()
            value = op.op(*var_values)  # 前向计算
            output_var = op.get_output_var()  # 赋值给输出变量,后续的op会使用
            output_var.set_value(value)
            # 为了打印输出，可以省略
            vars_list = op.get_input_vars()
            names = []
            for input_var in vars_list:
                names.append(input_var.name)
            print(str(names) + " -> " + op.name + " -> " + output_var.name)
        # 返回最后的输出 assume the last variable is output
        return self.list_variable[-1].value

    # 只有有一次前向调用后再调用此方法
    # delta_in 是后层回传梯度
    def build_backward_graph(self, delta_in):
        print(" ---- 开始反向计算 ----")
        self.zero_grads()  # 清空所有梯度
        name_stack = [("delta_in", delta_in, "y")]  # (计算节点名, 梯度值, 输出变量名)
        while len(name_stack) > 0:
            op_delta_var = name_stack.pop()
            op_name, delta_value, var_name = op_delta_var
            op_node = self.get_op_by_output_name(var_name)
            if op_node is None:  # 前面没有计算节点了，leaf var node
                var = self.get_var_by_name(var_name)
                print(op_name + " -> " + var_name + "[leaf]")
                if var is not None and var.is_constant == False:  # 不是常数
                    var.set_grad(delta_value)
                continue
            delta_out = op_node.op.backward(delta_value)
            input_vars = op_node.get_input_vars()
            flag = False  # 形状是否一致的标记，不一致时需要规约求和
            next_name = []
            if isinstance(delta_out, tuple) == True:
                assert(len(delta_out) == len(input_vars))
                for i, var in enumerate(input_vars):
                    name_stack.append((op_node.name, delta_out[i], var.name))
                    flag = self.__helper_check_shape(delta_value, delta_out[i])
                    next_name.append(var.name)
            else:
                name_stack.append((op_node.name, delta_out, input_vars[0].name))
                flag = self.__helper_check_shape(delta_value, delta_out)
                next_name.append(input_vars[0].name)
            if flag == True:
                print(var_name + " -> " + op_node.name + " -> " + str(next_name)  + "***")
            else:
                print(var_name + " -> " + op_node.name + " -> " + str(next_name))
            
        # 输出 x.grad
        var = self.get_var_by_name("x")
        return var.grad

    def __helper_check_shape(self, delta_in, delta_out):
        if isinstance(delta_in, np.ndarray) and isinstance(delta_out, np.ndarray):
            # shape相同的话则不处理
            # shape不同的话，必须列数相同
            # 输出的尺寸比输入的尺寸大
            if delta_out.shape != delta_in.shape and \
               delta_out.shape[1] == delta_in.shape[1] and \
               delta_out.shape[0] < delta_in.shape[0]: 
                return True
        return False
        

    def __str__(self):
        names = ""
        for op in self.list_operator:
            names += op.__str__() + "\r\n"
        return names


class my_model(Module):
    def __init__(self):
        super().__init__()
        self.mean_1 = layer.Mean()
        self.sub = layer.Sub()
        self.square = layer.Square()
        self.mean_2 = layer.Mean()
        self.add = layer.Add()
        self.sqrt = layer.Sqrt()
        self.div = layer.Div()

    def forward(self, x):
        self.x = x
        self.mu = self.mean_1(x)
        self.A = self.sub(x, self.mu)
        self.B = self.square(self.A)
        self.sigma = self.mean_2(self.B)
        self.eps = 1e-5
        self.C = self.add(self.sigma, self.eps)
        self.D = self.sqrt(self.C)
        self.y = self.div(self.A, self.D)
        return self.y

    # 根据第12章中的公式推导而来，用于做参考，验证自动反向梯度的正确性
    def backward_manual(self, dz):
        m = self.x.shape[0]
        a = dz / self.D
        b = np.sum(dz/self.D, axis=0, keepdims=True) / m
        c = np.sum(np.sum(dz * self.A, axis=0, keepdims=True) * self.A, axis=0,keepdims=True) /(m*m*np.power(self.D, 3))
        d = np.sum(dz * self.A, axis=0, keepdims=True) * self.A / (m * np.power(self.D, 3))
        dx = a - b + c - d
        return dx

    # 建立计算图
    def build_graph(self):
        # 图
        graph: compute_graph = compute_graph()
        # 变量节点
        var_x: variable_node = graph.add_variable(self.x, "x")
        var_mu: variable_node = graph.add_variable(self.mu, "mu")
        var_sigma: variable_node = graph.add_variable(self.sigma, "sigma")
        var_eps: variable_node = graph.add_variable(self.eps, "eps", is_constant=True)
        var_A: variable_node = graph.add_variable(self.A, "A")
        var_B: variable_node = graph.add_variable(self.B, "B")
        var_C: variable_node = graph.add_variable(self.C, "C")
        var_D: variable_node = graph.add_variable(self.D, "D")
        var_y: variable_node = graph.add_variable(self.y, "y")
        # 计算节点，这里是手工加入以展示过程，实际工程实践中是利用python语法树自动加入
        mean_1: compute_node = compute_node(self.mean_1, "mean_1", [var_x], var_mu)
        graph.add_operator(mean_1)
        sub: compute_node = compute_node(self.sub, "sub", [var_x, var_mu], var_A)
        graph.add_operator(sub)
        square: compute_node = compute_node(self.square, "square", [var_A], var_B)
        graph.add_operator(square)
        mean_2: compute_node = compute_node(self.mean_2, "mean_2", [var_B], var_sigma)
        graph.add_operator(mean_2)
        add: compute_node = compute_node(self.add, "add", [var_sigma, var_eps], var_C)
        graph.add_operator(add)
        sqrt: compute_node = compute_node(self.sqrt, "sqrt", [var_C], var_D)
        graph.add_operator(sqrt)
        div: compute_node = compute_node(self.div, "div", [var_A, var_D], var_y)
        graph.add_operator(div)

        return graph

def test_my_bn1d(x):
    bn = BatchNorm1d.BatchNorm1d(x.shape[1])
    z = bn.forward(x)
    dz = z + 1
    dx = bn.backward(dz)
    return z, dx    

import torch

def test_torch(batch, num_features):
    x = torch.randn(batch, num_features, requires_grad=True)
    bn = torch.nn.BatchNorm1d(num_features)
    z = bn(x)
    dz = z + 1
    z.backward(dz, retain_graph=True)
    return x.detach().numpy(), z.detach().numpy(), x.grad.detach().numpy()

if __name__=="__main__":
    num_features = 3
    batch = 4

    x, z_torch, dx_torch = test_torch(batch, num_features)
    print(" ---------- torch ---------")
    #print("x:", x)
    #print("z:", z)
    #x = np.random.rand(batch, num_features)
    z_bn1d, dx_bn1d = test_my_bn1d(x)
    print(" ---------- bn1d ---------")
    #print("x:", x)
    #print("z_bn1d:", z_bn1d)

    print(" ---------- manual ---------")
    model = my_model()
    z_manual = model.forward(x)
    #print("z_manual:", z_manual)
    delta = z_manual + 1
    dx_manual = model.backward_manual(delta)

    print(" ---------- autograd ---------")
    graph = model.build_graph()
    z_autograd = graph.build_forward_graph(x)
    print("正向图计算:", np.allclose(z_bn1d, z_torch))
    print("正向图计算:", np.allclose(z_manual, z_torch))
    print("正向图计算:", np.allclose(z_torch, z_autograd))
    # print(graph)
    delta = z_autograd + 1
    dx_autograd = graph.build_backward_graph(delta)

    print("dx_torch:\n", dx_torch)
    print("dx_BN1d:\n", dx_bn1d)
    print("dx_manual:\n", dx_manual)
    print("dx_autograd:\n", dx_autograd)

    print("反向图计算:", np.allclose(dx_bn1d, dx_torch, atol=1e-6, rtol=1e-7))
    print("反向图计算:", np.allclose(dx_torch, dx_autograd, atol=1e-6, rtol=1e-7))
    print("反向图计算:", np.allclose(dx_torch, dx_manual, atol=1e-6, rtol=1e-7))

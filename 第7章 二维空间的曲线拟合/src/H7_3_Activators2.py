# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from common.Activators import Relu, LeakyRelu, Sigmoid, Softplus, Tanh, Elu, GELU

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False
plt.rc('font', size=12)

def Draw(start, end, func, title):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    da = func.backward(z, a)

    plt.plot(z,a, linestyle="solid", label="函数")
    plt.plot(z,da, linestyle="dashed", label="导数")
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.xlabel("输入 : $z$")
    plt.ylabel("输出 : $a$")
    plt.show()

if __name__ == '__main__':
    # Draw(-5, 5, Relu(), "Relu")
    # Draw(-4, 4, Elu(0.8), "ELU")
    # Draw(-5, 5, LeakyRelu(0.1), "Leaky Relu")
    # Draw(-5, 5, Softplus(), "Softplus")
    Draw(-5, 5, GELU(), "GELU")

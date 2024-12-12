
import numpy as np
import matplotlib.pyplot as plt
import common.Optimizers as Optimizers

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=12)

def function(x, y):
    z = 0.15 * np.square(x) + 3 * np.square(y)
    return z

def func_derivative(w):
    return np.array([0.3 * w[0,0], 6 * w[0,1]])

def draw_loss(W4, labels, title):
    count = 100
    x = np.linspace(-10, 10, count)
    y = np.linspace(-10, 10, count)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    fig = plt.figure(figsize=(8,8))
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax = fig.add_subplot(2,2, idx+1)
            ax.contour(X, Y, Z, levels=(0.5,1,2,3,4,5,7,9,12,15,20,30), colors='lightgray')
            ax.set_ylim(-5,5)
            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")
            ax.plot(W4[idx][0,:],W4[idx][1,:], marker='.', label=str(labels[idx]))
            ax.legend()
    plt.suptitle(title)
    plt.show()

def run_lrs(opt_name, start_points, lrs, param = None):
    assert(start_points.shape[0] == len(lrs))
    W4 = []
    for idx in range(len(lrs)):
        optimizer = Optimizers.Optimizer.create_optimizer(opt_name, param)
        w = start_points[idx]
        lr = lrs[idx]
        W0 = []
        W1 = []
        for i in range(20):
            print(w)
            W0.append(w[0,0])
            W1.append(w[0,1])
            dw = func_derivative(w)
            w = optimizer.update(lr, w, dw)
        W = np.vstack((np.array(W0), np.array(W1)))
        W4.append(W)
    title = opt_name+"(%.2f)"%param if param is not None else opt_name
    draw_loss(W4, lrs, title)

def run_params(opt_name, start_points, params, lr, title):
    assert(start_points.shape[0] == len(params))
    W4 = []
    for idx in range(len(params)):
        optimizer = Optimizers.Optimizer.create_optimizer(opt_name, params[idx])
        w = start_points[idx]
        W0 = []
        W1 = []
        for i in range(20):
            print(w)
            W0.append(w[0,0])
            W1.append(w[0,1])
            dw = func_derivative(w)
            w = optimizer.update(lr, w, dw)
        W = np.vstack((np.array(W0), np.array(W1)))
        W4.append(W)
    draw_loss(W4, params, title)


if __name__=="__main__":

    start_points = np.array([[[-9, 4]],[[9, 4]],[[-9, -4]],[[9, -4]]])
    learning_rate = [0.1, 0.2, 0.3, 0.4]
    run_lrs("SGD", start_points, learning_rate)

    # name = "Momentum"
    # learning_rate = [0.1, 0.2, 0.3, 0.4]
    # run_lrs(name, start_points, learning_rate, param=0.5)

    name = "Momentum"
    learning_rate = 0.1
    params = [0.3, 0.5, 0.7, 0.9]
    run_params(name, start_points, params, learning_rate, "Momentum(lr=" + str(learning_rate) + ")")

    # name = "AdaGrad"
    # learning_rate = [0.5, 1.0, 1.5, 2.0]
    # #run_lrs(name, start_points, learning_rate)

    # learning_rate = [0.1, 0.2, 0.3, 0.4]
    # rmsprop = Optimizers.RMSProp(0.9)
    # #run_lrs(rmsprop, start_points, learning_rate, "RMSProp")

    # name = "RMSProp"
    # learning_rate = 0.1
    # params = [0.1, 0.3, 0.5, 0.7]
    # #run_params(name, start_points, params, learning_rate, "RMSProp(lr=" + str(learning_rate) + ")")

    # learning_rate = [0.1, 0.3, 0.5, 0.7]
    # #run_lrs("Adam", start_points, learning_rate)


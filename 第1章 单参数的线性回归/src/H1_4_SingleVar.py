import math

def func1():
    x_0 = 1.0
    eta = 0.1
    for i in range(100):
        x_1 = x_0 - eta * 2 * (x_0 - 2)
        x_0 = x_1
    print(x_0)

def func2(X, Y):
    for i in range(100):
        print("-----%d------"%i)
        print("X =",X)
        A = X + 1
        B = A * A
        C = math.log(B)
        D = math.sqrt(C)
        E = D - 1
        print("E =",E)
        X = X - 1 * A * (E-Y)/(B*math.sqrt(C))
        
def draw_fun():
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0.5, 3.5, 100)
    y = (x-2)**2+1
    plt.plot(x, y)
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.show()


if __name__=="__main__":
    draw_fun()
    func1()
    func2(2, 0.5)
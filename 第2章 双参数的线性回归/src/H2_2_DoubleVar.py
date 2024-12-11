
import numpy as np
import matplotlib.pyplot as plt

def target_function(x,y):
    J = x**2 + 2*np.sin(y)**2
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x, 4*np.sin(y)*np.cos(y)])

def show_3d_surface(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
 
    u = np.linspace(-2, 2, 100)
    v = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j]**2 + 2*np.sin(Y[i, j])**2

    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x,y,z, linewidth=3, color='black')
    plt.show()

if __name__ == '__main__':
    theta = np.array([2, 1.5])
    eta = 0.1
    error = 1e-2

    X = []
    Y = []
    Z = []
    for i in range(100):
        print(theta)
        x=theta[0]
        y=theta[1]
        z=target_function(x,y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("%d: x=%f, y=%f, z=%f" %(i,x,y,z))
        d_theta = derivative_function(theta)
        #print("",d_theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X,Y,Z)

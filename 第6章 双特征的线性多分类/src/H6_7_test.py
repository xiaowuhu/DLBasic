
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(0, 1, size=(128))
x2 = x1[0:64]

def func(x):
    return 0.5 * x

z1 = func(x1)
m1 = np.mean(np.square(z1))
#x3 = np.vstack((x1,x2))
z3 = func(x2)
m3 = np.mean(np.square(z3))
np.set_printoptions(precision=4)
#print(x1)
print(np.var(x1))
print(m1)
print("-------")
#print(x2)
print(np.var(x2))
print(m3)

plt.scatter(x1, [0]*len(x1), marker='o')
plt.scatter(x2, [0]*len(x2), marker='x')
plt.show()

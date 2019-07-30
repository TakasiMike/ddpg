import numpy as np
from numpy.linalg import inv


x = np.empty(1000)
y = np.empty(1000)

x[0] = 0
x[1] = 0
delta = np.empty(shape=[2, 1])



def f_1(x, y):
    return np.exp(-x) - y


def f_2(x, y):
    return x - y ** 2 - 3 * y


def f_1x(x, y, h):
    return (f_1(x + h, y) - f_1(x, y)) / h


def f_1y(x, y, h):
    return (f_1(x, y + h) - f_1(x, y)) / h


def f_2x(x, y, h):
    return (f_2(x + h, y) - f_2(x, y)) / h


def f_2y(x, y, h):
    return (f_2(x, y + h) - f_2(x, y)) / h


def Jacobian_matrix(x, y, h):
    return np.array([[f_1x(x, y, h), f_1y(x, y, h)], [f_2x(x, y, h), f_2y(x, y, h)]])


for j in range(10):
    for i in range(1, 2):
        f = np.array([f_1(x[i-1], y[i-1]), f_2(x[i-1], y[i-1])])
        f_tr = np.transpose(f)
        delta = np.dot(inv(Jacobian_matrix(x[i-1], y[i-1], 0.0000001)), - f_tr)
        x[i] = x[i-1] + delta[i]
        x[i-1] = x[i]

        print(x[i])



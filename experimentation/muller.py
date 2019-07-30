import numpy as np


def f(x):
    return x ** 3 + x ** 2 + x + 1

x1 = 5
x2 = 0
x3 = -5
tol = 10 ** -5
max_iter = 1000
fx1 = f(x1)
fx2 = f(x2)
fx3 = f(x3)
x = np.empty(shape=[max_iter, 1], dtype='complex128')
x[0] = x1
x[1] = x2
x[2] = x3

iterations = 0
for i in range(2, max_iter):
    iterations += 1
    q = (x[i] - x[i-1]) / (x[i-1] - x[i-2])
    A = (q * f(x[i])) - (q * (q + 1) * f(x[i-1])) + q * q * f(x[i-2])
    B = (1 + 2 * q) * f(x[i]) - ((1 + q) ** 2) * f(x[i-1]) + q * q * f(x[i-2])
    C = (1 + q) * f(x[i])
    if B < 0:
        x[i+1] = x[i] - (x[i] - x[i-1]) * ((2 * C) / (B - np.sqrt((B ** 2) - 4 * A * C)))
    else:
        x[i + 1] = x[i] - (x[i] - x[i - 1]) * ((2 * C) / (B + np.sqrt((B ** 2) - 4 * A * C)))

    if np.abs(f(x[i+1])) < tol:
        iterations = iterations
        print('The root of the function is :', x[i+1])
        print('It was found after ', iterations, 'iterations')
        break














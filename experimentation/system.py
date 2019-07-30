import numpy as np


def f(x):
    return x ** 3 - 3 * x ** 2 - 6 * x + 8

def f_prime(x):
    return 3 * x ** 2 - 6 * x - 6

def f_double_prime(x):
    return 6 * x  - 6

x0 = float(input('Insert initial guess :'))
max_iter = 1000
tol = 10 ** - 8
x = np.empty(max_iter, dtype='complex128')
x[0] = x0

iter = 0
for i in range(max_iter):
    iter += 1
    if f_prime(x[i]) < 0:
        x[i+1] = x[i] - (2 * f(x[i])) / (f_prime(x[i]) - np.sqrt((f_prime(x[i]) ** 2) - 2 * f(x[i]) * f_double_prime(x[i])))
    else:
        x[i + 1] = x[i] - (2 * f(x[i])) / (
                    f_prime(x[i]) + np.sqrt((f_prime(x[i]) ** 2) - 2 * f(x[i]) * f_double_prime(x[i])))

    if iter == max_iter:
        print('The method could not converge')
        break


    if abs(f(x[i+1])) < tol:
        iter = iter
        print('One root of the function is ', x[i+1])
        print('It was found after ', iter, 'iterations')
        break













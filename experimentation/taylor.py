import math
from matplotlib import pyplot as plt


def taylor_sine(n, x):
    h = 0
    for i in range(1, n):
        h += ((-1) ** (i - 1)) * ((x ** ((2 * i) - 1)) / (math.factorial((2 * i) - 1)))
    return h

def error(x):
    js = []
    err = []
    for j in range(10000):

        diff = abs(math.sin(x) - taylor_sine(j, x))
        js.append(j)
        err.append(diff)
        print(j, diff)
        # plt.figure(1)
        # plt.plot(js, err)
        # plt.xlabel('Number of iterations')
        # plt.ylabel('Absolute Error')
        # plt.show()
        # plt.pause(0.1)
        if diff <= 10 ** -8:
            print('Done')
            break
    plt.figure(1)
    plt.plot(js, err)
    plt.xlabel('Number of iterations')
    plt.ylabel('Absolute Error')
    plt.show()
    plt.pause(0.1)

error(0.5)


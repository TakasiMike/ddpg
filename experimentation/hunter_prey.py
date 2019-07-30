from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np

sigma = 10
b = 8 / 3
r = 28

def lifeanddeath(state, t):
    x, y, z = state
    dxdt = -sigma * x + sigma * y
    dydt = - x * z + r * x - y
    dzdt = x * y - b * z

    return [dxdt, dydt, dzdt]

t = np.linspace(0, 60)

init_cond = [10, 10, 10]
solution = odeint(lifeanddeath, init_cond, t)
print(solution)

for i in range(1, 50):
    plt.clf()
    plt.ion()
    plt.plot(t, solution)
    plt.show()
    plt.pause(0.001)









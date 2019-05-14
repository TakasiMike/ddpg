import numpy as np
from scipy.integrate import odeint
import scipy as sc
import matplotlib.pyplot as plt


def model(f, t):
    k = 1/0.6
    dydt = k * f
    return dydt


def eval_solution(fun):
    k = 1/0.6
    return sc.exp(k) * fun


init_cond = np.empty([200, 1], dtype=np.float32)
np.append(init_cond, 0.01)

for t in range(200):
    time_interval = np.linspace(t, t + 1)
    y0 = init_cond[t]
    solution = odeint(model, y0, time_interval)
    np.append(init_cond, eval_solution(solution))

    plt.plot(time_interval, solution)
    plt.figure(1)

plt.show()



























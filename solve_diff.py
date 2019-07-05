from scipy.integrate import odeint
import numpy as np
from pylab import *
import matplotlib.animation as animation
import math
#
#
# def BoatFishSystem(state, t):
#     fish, boat = state
#     d_fish = fish * (2 - boat - fish)
#     d_boat = -boat * (1 - 1.5 * fish)
#     return [d_fish, d_boat]
#
#
# t = np.linspace(0, 20)
# init_state = [1, 1]
# state = odeint(BoatFishSystem, init_state, t)
# print(state)
# fish = state[49][0]
# print(fish)
# boat = state[49][1]
# print(boat)
#
# fig = figure()
# xlabel('number of fish')
# ylabel('number of boats')
# plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
#
# def animate(i):
#     plot(state[0:i, 0], state[0:i, 1], 'b-')
#
#
# ani = animation.FuncAnimation(fig, animate, interval=1)
# show()

# Παράμετροι
V = 100
UA = 20000
density = 1000
Cp = 4.2
minus_DH = 596619
k0 = 6.85 * (10 ** 11)
E = 76534.704
R = 8.314
T_in = 300
Ca_in = 1
T_j = 250
F = 20

def equations(state, t):
    Ca, T = state
    d_Ca = (F / V) * (Ca_in - Ca) - 2 * k0 * math.exp(- E / (R * T)) * (Ca ** 2)
    d_T = (F / V) * (T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * math.exp(- E / (R * T)) * (Ca ** 2) - (UA / (V * density * Cp)) * (T - T_j)
    return [d_Ca, d_T]


time = np.linspace(0, 100)

init_cond = [Ca_in, T_in]
solution = odeint(equations, init_cond, time)
print(solution)
print(solution[49][0])






















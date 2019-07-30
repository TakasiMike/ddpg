from scipy.optimize import fsolve
import numpy as np


V = 100
UA = 20000
density = 1000
Cp = 4.2
minus_DH = 596619
k0 = 6.85 * (10 ** 11)
E = 76534.704
R = 8.314
T_in = 276
Ca_in = 1
T_j = 251
f = 20.5


def equations(z):
    Ca = z[0]
    T = z[1]
    F = np.empty(2)
    F[0] = ((f / V) * (Ca_in - Ca)) - 2 * k0 * np.exp(- E / (R * T)) * (Ca ** 2)
    F[1] = (f / V) * (T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * np.exp(- E / (R * T)) * (Ca ** 2) - ((UA / (V * density * Cp)) * (T - T_j))
    return F


init_cond = np.array([0.07, 370])
z = fsolve(equations, init_cond)
print(z)
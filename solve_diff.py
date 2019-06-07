from scipy.integrate import odeint
from numpy import arange
import math

# Παράμετροι
V = 100
UA = 20000
density = 1000
Cp = 4.2
minus_DH = 596619
k0 = 6.85 * (10 ** 11)
E = 76534.704
R = 8.314
T_in = 275
Ca_in = 1

def equations(state, t):
    Ca, T = state
    d_Ca = (F / V)*(Ca_in - Ca) - 2 * k0 * math.exp(E /(R * T)) * (Ca ** 2)
    d_T = (F / V)*(T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * math.exp(E / (R * T)) * (Ca ** 2) - \
          (UA / (V * density * Cp))*(T - T_j)
    return [d_Ca, d_T]


























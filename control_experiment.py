import control
import numpy as np
import math


T = np.linspace(0, 1)
action = 30

init_cond = 0

def output(system):
    return control.forced_response(system, T, U=action, X0=init_cond)[1]


g = control.tf([0, 0, 0, 0, 0.05], [1, -0.6, 0, 0, 0])
print(g)
sys = control.tf2ss(g)
print(sys)
print(output(sys))
print(abs(2 - 4))







import control
import numpy as np
import math


T = np.linspace(0, 1)
action = 30

init_cond = 0

def output(system):
    return control.forced_response(system, T, U=action, X0=init_cond)


g = control.tf([0.05, 0], [-0.6, 1])
print(g)
sys = control.tf2ss(g)
print(sys)
print(output(sys))







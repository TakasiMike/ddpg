import control
import numpy as np
import math


T = np.linspace(0, 1)
action = 2

init_cond = math.log(10)


def output(system):
    return control.forced_response(system, T, U=action, X0=init_cond)[1]


g = control.tf([0.05, 0], [-0.6, 1])
print(g)
sys = control.tf2ss(g)
print(sys)
print(output(sys))







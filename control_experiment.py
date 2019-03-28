import control
import numpy as np


T = np.linspace(0, 1, 100)


def output(system):
    return control.forced_response(system, T)[1][99]


g = control.tf([0.05, 0], [-0.6, 1])
print(g)
sys = control.tf2ss(g)
print(sys)
print(output(sys))






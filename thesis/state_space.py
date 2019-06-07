import numpy as np
from actor_network import ActorNet
import control


class StateSpace:
    def __init__(self, no_states, no_actions):
        self.no_states = no_states
        self.no_actions = no_actions
        self.input = ActorNet.create_actor_net(no_states, no_actions)

    def output(self, act):
        g = control.tf([0.05, 0], [-0.6, 1])
        print(g)

        sys = control.tf2ss(g)
        print(sys)

        time_interval = np.linspace(0, 1, 100)

        return control.forced_response(sys, time_interval, act)



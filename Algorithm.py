from ddpg import ddpg
from state_space import StateSpace
import numpy as np

steps = 200
y_set = np.random.uniform(0, 1)


def main():
    agent = ddpg
    total_reward = 0
    num_of_states = 2
    num_of_actions = 1
    reward = np.array([0])

    # For Loop του αλγορίθμου για ένα επισόδειο
    for t in range(steps):
        s_t = StateSpace.output()







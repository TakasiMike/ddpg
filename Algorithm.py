from ddpg import DDPG
import numpy as np
import control
import matplotlib.pyplot as plt

steps = 100
y_set = np.random.uniform(0, 1)
eps = 0.001  # Tolerance του reward function
c = 10  # reward value

# Το περιβάλλον του προβλήματος
g = control.tf([0.05, 0], [-0.6, 1])
sys = control.tf2ss(g)


def reward(state):
    if abs(state - y_set) < eps:
        return c
    else:
        return -np.power(abs(state - y_set), 2)


def execute():
    agent = DDPG(2, 1)
    reward_per_episode = 0

    # For Loop του αλγορίθμου για ένα επισόδειο
    for t in range(steps):
        if t == 1:
            current_moisture = 0.01
        else:
            current_moisture = agent.minibatches[0][0]  # y(t)

        current_state = [current_moisture, y_set]  # s
        action = agent.evaluate_actor(current_state)  # Δίνει το action , α(t)

        T = np.linspace(t, t + 1, 10)

        def output(system):
            return control.forced_response(system, T, U=action, X0=current_moisture)[1][9]

        next_moisture = output(sys)  # y(t+1)

        next_state = [next_moisture, y_set]  # s'
        current_reward = reward(next_moisture)  # r

        reward_per_episode += reward(next_moisture)  # Συνολικό reward
        agent.add_experience(current_state, next_state, action, current_reward)
        agent.model_train()


execute()
















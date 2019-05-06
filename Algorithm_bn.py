from ddpg_bn import DDPG
import numpy as np
# import matplotlib
import control
import tensorflow as tf
from RM import ReplayMemory
# matplotlib.use("TkAgg")

# import matplotlib.pyplot as plt

steps = 1000
episodes = 500
y_set = np.random.uniform(0, 1)
print(y_set)
eps = 0.01  # Tolerance του reward function
c = 100  # reward value

# Το περιβάλλον του προβλήματος
g = control.tf([0.05, 0], [-0.6, 1])
sys = control.tf2ss(g)
number_of_states = 2
number_of_actions = 1


def reward(state):
    if abs(state - y_set) < eps:
        return c
    else:
        return -np.power(abs(state - y_set), 2)


def output(system, T, U, init_cond):
    return control.forced_response(system, T, U, init_cond)[1][49]


def main():
    with tf.Graph().as_default():
        agent = DDPG(number_of_states, number_of_actions)
        reward_per_time_step = 0
        RM = ReplayMemory(100000)


        for e in range(episodes):
            print('Begin Episode number', e)
            # For Loop του αλγορίθμου για ένα επισόδειο
            for t in range(steps):
                if t == 0:
                    current_moisture = 0
                else:
                    current_moisture = RM.replay_memory[-1][0][0]  # y(t)
                    print(current_moisture)

                current_state = np.array([current_moisture, y_set]) # s
                # print(current_state)
                current_state_true = current_state.reshape(1, 2)
                action = agent.evaluate_actor(current_state_true)[0][0]  # Δίνει το action , α(t)
                # print(action)
                T = np.linspace(t, t + 1)

                next_moisture = output(sys, T, action, current_moisture)  # y(t+1)
                # print(next_moisture)

                next_state = np.array([next_moisture, y_set])  # s'

                current_reward = reward(agent.model_train(RM=RM)[-1])  # r
                # print("this thing =", agent.model_train(RM=RM)[-1])
                # print(current_reward)

                reward_per_time_step += current_reward  # Συνολικό reward
                # print(reward_per_time_step)

                RM.add_experience(current_state, next_state, action, current_reward)

                agent.model_train(RM=RM)


if __name__ == "__main__":
    main()

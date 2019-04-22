from ddpg import DDPG
import numpy as np
import control
import tensorflow as tf
import matplotlib.pyplot as plt

steps = 100
y_set = np.random.uniform(0, 1)
eps = 0.001  # Tolerance του reward function
c = 10  # reward value

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


        # For Loop του αλγορίθμου για ένα επισόδειο
        for t in range(steps):
            if t == 1:
                current_moisture = 0.01
            else:
                current_moisture = t  # y(t)

            current_state = np.array([current_moisture, y_set])  # s
            current_state_true = current_state.reshape(1, 2)
            action = agent.evaluate_actor(current_state_true)[0][0]  # Δίνει το action , α(t)


            T = np.linspace(t, t + 1)

            next_moisture = output(sys, T, action, current_moisture)  # y(t+1)

            next_state = np.array([next_moisture, y_set])  # s'

            current_reward = reward(next_moisture)  # r   #Αυτό πρέπει να είναι λάθος

            reward_per_time_step += current_reward  # Συνολικό reward
            agent.add_experience(current_state, next_state, action, current_reward)
            agent.model_train()


if __name__ == "__main__":
    main()
















from ddpg_bn import DDPG
import numpy as np
from ou_noise import OUNoise
import matplotlib.pyplot as plt
import control
import tensorflow as tf
# from RM import ReplayMemory
# matplotlib.use("TkAgg")
import math



steps = 300
episodes = 5000
# y_set = np.random.uniform(0, 1)
y_set = 1.5
print(y_set)
eps = 0.01  # Tolerance του reward function
c = 100  # reward value
error = 0.005

# Το περιβάλλον του προβλήματος
# g = control.tf([-0.05, 0], [-0.6, 1])
g = control.tf([0, 0, 0, 0, 0.05], [1, -0.6, 0, 0, 0])
sys = control.tf2ss(g)
number_of_states = 2
number_of_actions = 1


# def reward(state):
#     if abs(state - y_set) < eps and state < y_set + error:
#         return 0
#     else:
#         return -10

def reward(state, state_next):
    if abs(state_next - y_set) < abs(state - y_set) and state_next < y_set + error and abs(state - y_set) < eps:
        return 100 / (math.sqrt(abs(state_next - y_set)) + 1)
    else:
        return -5

# def reward(state, state_next):
#     if abs(state_next - y_set) < abs(state - y_set):
#         return 0
#     else:
#         return -1

def output(system, T, U, init_cond):
    return control.forced_response(system, T, U, init_cond)[1][49]


def main():
    with tf.Graph().as_default():
        agent = DDPG(number_of_states, number_of_actions)
        exploration_noise = OUNoise(number_of_actions)
        reward_per_episode = 0
        total_reward = 0
        # c_r = np.array([0])
        this = 0


        for e in range(episodes):
            print('Begin Episode number', e)
            # For Loop του αλγορίθμου για ένα επισόδειο
            for t in range(steps):
                if t == 0:
                    current_moisture = 0
                else:
                    current_moisture = agent.replay_memory[-1][1][0]  # y(t)
                    # c_r = np.append(c_r, current_moisture)
                    # np.savetxt('current_moisture.txt', c_r, newline="\n")
                print('c_moist ' + str(current_moisture))

                # print('set_moist ' + str(y_set))

                current_state = np.array([current_moisture, y_set])  # s
                # print(current_state)
                current_state_true = current_state.reshape(1, 2)
                action = agent.evaluate_actor(current_state_true)  # Δίνει το action , α(t)
                noise = exploration_noise.noise()  # OU-Noise Ν
                action_true = action[0][0] + noise[0]
                # print(action[0][0])
                T = np.linspace(0, 1)
                # T = np.linspace(t, t + 1)

                next_moisture = output(sys, T, action_true, current_moisture)  # y(t+1)
                # next_moisture = output(g, T, action, current_moisture)
                # print('n_moist ' + str(next_moisture))
                # print(next_moisture)

                next_state = np.array([next_moisture, y_set])  # s'

                # current_reward = reward(next_moisture)  # r
                # print('rew ' + str(current_reward))
                current_reward = reward(current_moisture, next_moisture)
                # print("this thing =", agent.model_train(RM=RM)[-1])
                # print(current_reward)
                agent.add_experience(current_state, next_state, current_reward, action_true)

                if this > 20000:
                    agent.model_train()
                reward_per_episode += current_reward  # Συνολικό reward
                this += 1
                if t == steps - 1:
                    exploration_noise.reset()

                if t > 20 and abs((agent.replay_memory[-1][1][0]) - y_set) < eps and abs((agent.replay_memory[-2][1][0]) - y_set) < eps and abs((agent.replay_memory[-3][1][0]) - y_set) < eps and abs((agent.replay_memory[-4][1][0]) - y_set) < eps and abs((agent.replay_memory[-5][1][0]) - y_set) < eps:
                        break
            total_reward += reward_per_episode
            # print(total_reward)


if __name__ == "__main__":
    main()

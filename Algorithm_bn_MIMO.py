from ddpg_bn_MIMO import DDPG
import numpy as np
from ou_noise import OUNoise
import tensorflow as tf

from scipy.integrate import odeint

# sess = tf.Session()
# saver = tf.train.import_meta_graph('DDPG_MIMO-1000.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./'))


# Παράμετροι
V = 100
UA = 20000
density = 1000
Cp = 4.2
minus_DH = 596619
k0 = 6.85 * (10 ** 11)
E = 76534.704
R = 8.314
T_in = 275
Ca_in = 1
# T_j = 250
# F = 20


# def equations(state, t):
#
#     Ca, T = state
#     d_Ca = (F / V)*(Ca_in - Ca) - 2 * k0 * math.exp(E / (R * T)) * (Ca ** 2)
#     d_T = (F / V)*(T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * math.exp(E / (R * T)) * (Ca ** 2) - \
#           (UA / (V * density * Cp))*(T - T_j)
#     return [d_Ca, d_T]


steps = 300
episodes = 3000
C_set = 0.07
T_set = 376
print(C_set)
print(T_set)
eps_1 = 0.003
eps_2 = 1
c = 1000  # reward value
error = 0.005


number_of_states = 4
number_of_actions = 2

#
# def reward(state_1, state_2):
#     if abs(state_1 - C_set) < eps_1 and abs(state_2 - T_set) < eps_2:
#         return c
#     else:
#         return -abs(state_1 - C_set) - abs(state_2 - T_set)

# def reward(state1, state1_next, state2, state2_next):
#     if abs(state1_next - C_set) < abs(state1 - C_set) and abs(state2_next - T_set) < abs(state2 - T_set):
#         return 100 / (math.sqrt(abs(state1_next - C_set)) + math.sqrt(abs(state2_next - T_set)) + 1)
#     else:
#         return -1


# def reward(state1, state1_next, state2, state2_next):
#     if abs(state1_next - C_set) < abs(state1 - C_set) and abs(state2_next - T_set) < abs(state2 - T_set):
#         return 0
#     else:
#         return -1

# def reward(state1, state1_next, state2, state2_next, temp_of_cool):
#     if abs(state1_next - C_set) < abs(state1 - C_set) and abs(state2_next - T_set) < abs(state2 - T_set):
#         return 1
#     elif abs(state1_next - C_set) < eps_1 and abs(state2_next - T_set) < eps_2:
#         return 100
#     elif state1_next < C_set - 0.005 or state2_next > T_set + 1:
#         return -20
#     else:
#         return -10


def reward(state1, state1_next, state2, state2_next, act_T_t):
    if abs(state1_next - C_set) < abs(state1 - C_set) and abs(state2_next - T_set) < abs(state2 - T_set):
        return 1
    elif abs(state1_next - C_set) < eps_1 and abs(state2_next - T_set) < eps_2:
        return 200
    elif state1_next < C_set - 0.005 or state2_next > T_set + 1:
        return -20
    elif act_T_t > 300:
        return -100
    # elif abs(act_T_t - act_T_t_1) > 2:
    #     return -15
    else:
        return -10


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
                    current_Ca = Ca_in
                    current_T = T_in
                else:
                    current_Ca = agent.replay_memory[-1][1][0]  # y(t)
                    current_T = agent.replay_memory[-1][1][2]
                    # c_r = np.append(c_r, current_moisture)

                print('Ca ' + str(current_Ca))
                print('T ' + str(current_T))

                current_state = np.array([current_Ca, C_set, current_T, T_set], dtype=np.float64)  # s
                # print(current_state)
                current_state_true = current_state.reshape(1, 4)
                noise = exploration_noise.noise()
                # if t == 0:
                #     action = [F, T_j]
                #     action_true_F = F + noise[0]
                #     action_true_T = T_j + noise[1]
                # else:
                action = (agent.evaluate_actor(current_state_true))[0]  # Δίνει το action , α(t)
                action_true_F = action[0] + noise[0]
                print('Flow ' + str(action_true_F))
                # action_true_F = action[0]
                # print('action_F ' + str(action_true_F))
                action_true_T = action[1] + noise[1]
                action_true = np.array([action_true_F, action_true_T])

                # action_true_T = action[1]
                print('Temp cool ' + str(action_true_T))

                Time = np.linspace(0, 1)
                init_cond = [current_Ca, current_T]

                def equations(state, t):

                    Ca, T = state
                    d_Ca = (action_true_F / V) * (Ca_in - Ca) - 2 * k0 * np.exp(- E / (R * T)) * (Ca ** 2)
                    d_T = (action_true_F / V) * (T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * np.exp(- E / (R * T)) * (Ca ** 2) - (UA / (V * density * Cp)) * (T - action_true_T)
                    return [d_Ca, d_T]

                solution = odeint(equations, init_cond, Time, mxstep=1000000)
                next_Ca = solution[49][0]  # Ca(t+1)
                next_T = solution[49][1]  # T(t+1)

                next_state = np.array([next_Ca, C_set, next_T, T_set], dtype=np.float64)  # s'
                # current_reward = reward(current_Ca, next_Ca, current_T, next_T, action_true_T, agent.replay_memory[-1][-1][1])
                current_reward = reward(current_Ca, next_Ca, current_T, next_T, action_true_T)
                # current_reward = reward(next_Ca, next_T)

                # print(current_reward)
                agent.add_experience(current_state, next_state, current_reward, action_true)

                if this > 10000:
                    agent.model_train()
                reward_per_episode += current_reward  # Συνολικό reward
                this += 1
                if t == steps - 1:
                    exploration_noise.reset()

                # if t > 20 and abs((agent.replay_memory[-1][1][0]) - C_set) < eps and abs((agent.replay_memory[-2][1][0]) - C_set) < eps and abs((agent.replay_memory[-3][1][0]) - C_set) < eps and abs((agent.replay_memory[-4][1][0]) - C_set) < eps and abs((agent.replay_memory[-5][1][0]) - C_set) < eps\
                #         and abs((agent.replay_memory[-1][1][2]) - T_set) < eps and abs((agent.replay_memory[-2][1][2]) - T_set) < eps and abs((agent.replay_memory[-3][1][2]) - T_set) < eps and abs((agent.replay_memory[-4][1][2]) - T_set) < eps and abs((agent.replay_memory[-5][1][2]) - T_set) < eps:
                #         break
                if t > 20 and abs((agent.replay_memory[-1][1][0]) - C_set) < eps_1 and abs((agent.replay_memory[-2][1][0]) - C_set) < eps_1 and abs((agent.replay_memory[-3][1][0]) - C_set) < eps_1 and abs((agent.replay_memory[-4][1][0]) - C_set) < eps_1 and abs((agent.replay_memory[-5][1][0]) - C_set) < eps_1\
                        and abs((agent.replay_memory[-1][1][2]) - T_set) < eps_2 and abs((agent.replay_memory[-2][1][2]) - T_set) < eps_2 and abs((agent.replay_memory[-3][1][2]) - T_set) < eps_2 and abs((agent.replay_memory[-4][1][2]) - T_set) < eps_2 and abs((agent.replay_memory[-5][1][2]) - T_set) < eps_2:
                        # with tf.Session() as sess:
                        #     sess = tf.InteractiveSession()
                        #     saver = tf.train.Saver()
                        #     saver.save(sess, 'ddpg_trained')
                        break

            total_reward += reward_per_episode
            # print(total_reward)



if __name__ == "__main__":
    main()
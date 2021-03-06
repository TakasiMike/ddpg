import numpy as np
from collections import deque
# from actor_network import ActorNet
# from critic_network import CriticNet
from actor_network_bn import ActorNet_bn
from critic_network_bn import CriticNet_bn
from grad_inverter_MIMO import grad_inverter
import random


RM_size = 10000
Batch_Size = 128
capacity = 10000

Gamma = 0.99  # Discount Factor

class DDPG:

    def __init__(self, num_of_states, num_of_actions):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.critic_net = CriticNet_bn(self.num_of_states, self.num_of_actions)
        self.actor_net = ActorNet_bn(self.num_of_states, self.num_of_actions)

        self.replay_memory = deque()

        # Intialize time step:
        self.time_step = 0
        self.counter = 0

        # Ορισμός μέγιστης και ελάχιστης δράσης και κάλεσμα του grad inverter
        action_F_max = 24
        action_F_min = 16
        action_Tj_max = 300
        action_Tj_min = 200
        action_min = np.array([action_F_min, action_Tj_min])
        action_max = np.array([action_F_max, action_Tj_max])
        action_bounds = [action_max, action_min]
        self.grad_inverter = grad_inverter(action_bounds)

    def evaluate_actor(self, state_now):
        return self.actor_net.evaluate_actor(state_now)

    # Συνάρτηση που βάζει ένα experience (s,a,r,s') στο RM
    def add_experience(self, current_state, next_state, reward, action):
        self.current_state = current_state
        self.next_state = next_state
        self.reward = reward
        self.action = action
        self.replay_memory.append((self.current_state, self.next_state, self.reward, self.action))
        self.time_step += 1
        if len(self.replay_memory) > capacity:
            self.replay_memory.popleft()
        # print(self.replay_memory)
        return self.replay_memory

    def minibatches(self):  # Επιλέγει ένα τυχαίο batch από την μνήμη μεγέθους batch_size

        batch = random.sample(self.replay_memory, Batch_Size)

        # State y(t)
        self.current_state_batch = [item[0] for item in batch]
        self.current_state_batch = np.array(self.current_state_batch)
        # print(self.current_state_batch.shape)
        # Next State y(t+1)
        self.next_state_batch = [item[1] for item in batch]
        self.next_state_batch = np.array(self.next_state_batch)
        # Reward r(t)
        self.reward_batch = [item[2] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        # print(self.reward_batch.shape)

        # Action π(t)
        self.action_batch = [item[3] for item in batch]
        self.action_batch = np.array(self.action_batch)
        # print(self.action_batch)
        self.action_batch = np.reshape(self.action_batch, [len(self.action_batch), self.num_of_actions])
        return batch



    # Συνάρτηση που θα εκπαιδεύει το μοντέλο
    def model_train(self):
        self.minibatches()

        self.next_action_batch = self.actor_net.evaluate_target_actor(self.next_state_batch)
        # Σχηματισμός του Q_t ^i (s',a',W)
        # print(self.next_action_batch.shape)
        # print(self.next_action_batch)
        # print(self.next_state_batch.shape)
        # print(self.next_state_batch)
        q_next = self.critic_net.evaluate_target_critic(self.next_state_batch, self.next_action_batch)
        # print(q_next)
        # Υπολογισμός του reward και προσθήκη του στο άδειο array παρακάτω
        self.y_i_batch = []
        for i in range(0, Batch_Size):
            # print(self.reward_batch[i])

            if i == Batch_Size:
                self.y_i_batch.append(self.reward_batch[i])

            else:
                self.y_i_batch.append(self.reward_batch[i] + Gamma*q_next[i][0])


        self.y_i_batch = np.array(self.y_i_batch)
        # print(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])


        # Update του critic (Βήμα 16 του αλγορίθμου):
        self.critic_net.train_critic(self.current_state_batch, self.action_batch, self.y_i_batch)

        # Update του actor:

        # Υπολογισμός ενός action από το δίκτυο του actor που θα χρησιμοποιηθεί για το gradient:
        action_for_gradient = self.evaluate_actor(self.current_state_batch)

        # Υπολογισμός του gradient inverter (Βήμα 17 του αλγορίθμου):
        self.dq_da = self.critic_net.compute_delQ_a(self.current_state_batch, action_for_gradient)

        self.dq_da = self.grad_inverter.inverter(self.dq_da, action_for_gradient)
        # print(self.dq_da)


        # Update του actor (Βήμα 19 του αλγορίθμου):

        self.actor_net.train_actor(self.current_state_batch, self.dq_da)

        # Τελικό update του target actor & critic
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        return self.y_i_batch

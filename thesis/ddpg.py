import numpy as np
from actor_network import ActorNet
from critic_network import CriticNet
from collections import deque
import random
from grad_inverter import Grad_Inverter


# Καθολικές Παράμετροι
RM_size = 100000
Batch_Size = 32
Gamma = 0.99  # Discount Factor
c = 10  # Reward Constant
y_set = 1  # Set point


class ddpg:

    def __init__(self, current_state, next_state, action, reward, done, current_state_batch,
                 next_state_batch, action_batch, reward_batch, done_batch, next_action_batch, y_i_batch, num_states,
                 num_actions, dq_da):
        self.num_states = num_states
        self.num_actions = num_actions
        self.critic_net = CriticNet(self.num_states, self.num_actions)
        self.actor_net = ActorNet(self.num_states, self.num_actions)
        self.current_state = current_state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.current_state_batch = current_state_batch
        self.next_state_batch = next_state_batch
        self.action_batch = action_batch
        self.reward_batch = reward_batch
        self.done_batch = done_batch
        self.next_action_batch = next_action_batch
        self.y_i_batch = y_i_batch
        self.dq_da = dq_da

        # Initialization του Replay Memory
        self.replay_memory = deque()
        self.time_step = 0

        # Ορισμός μέγιστης και ελάχιστης δράσης και κάλεσμα του grad inverter
        action_max = 35
        action_min = 0
        action_bounds = [action_max, action_min]
        self.grad_inverter = Grad_Inverter(action_bounds)

    # Συνάρτηση που βάζει ένα experience (s,a,r,s') στο RM
    def add_experience(self, current_state, next_state, action, reward, done):
        self.current_state = current_state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.replay_memory.append(self.current_state, self.next_state, self.action, self.reward)
        self.time_step += 1
        if len(self.replay_memory) > RM_size:
            self.replay_memory.popleft()

    def evaluate_actor(self, current_state):
        return self.actor_net.evaluate_actor(current_state)

    # Συνάρτηση που ορίζει ένα mini batch tuple (s,a,s',r)
    def minibatches(self):
        batch = random.sample(self.replay_memory, Batch_Size)
        # State y(t)
        self.current_state_batch = [item[0] for item in batch]
        self.current_state_batch = np.array(self.current_state_batch)
        # Next State y(t+1)
        self.next_state_batch = [item[1] for item in batch]
        self.next_state_batch = np.array(self.next_state_batch)
        # Reward r(t)
        self.reward_batch = [item[2] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        # Action π(t)
        self.action_batch = [item[3] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch, [len(self.action_batch), self.num_actions])
        # Done batch(t)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)

    def reward(self, i):
        if abs(self.y_i_batch[i] - y_set):
            return c
        else:
            return -np.power(abs(self.y_i_batch[i] - y_set), 2)

    # Συνάρτηση που θα εκπαιδεύει το μοντέλο
    def model_train(self):
        self.minibatches()
        # Σχηματισμός της επόμενης δράσης π(s',W)
        self.next_action_batch = self.actor_net.evaluate_target_actor(self.next_state_batch)
        # Σχηματισμός του Q_t ^i (s',a',W)
        q_next = self.critic_net.evaluate_target_network(self.next_state_batch, self.next_action_batch)
        # Υπολογισμός του reward και προσθήκη του στο άδειο array παρακάτω
        self.y_i_batch = []
        for i in range(0, Batch_Size):
            if i == Batch_Size:
                self.y_i_batch.append(self.reward())
            else:
                self.y_i_batch.append(self.reward() + Gamma*q_next[i][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])

        # Update του critic (Βήμα 16 του αλγορίθμου):
        self.critic_net.train_critic(self.current_state_batch, self.action_batch, self.y_i_batch)

        # Update του actor:

        # Υπολογισμός ενός action από το δίκτυο του actor που θα χρησιμοποιηθεί για το gradient:
        action_for_gradient = self.actor_net.evaluate_actor(self.current_state_batch)

        # Υπολογισμός του gradient inverter (Βήμα 17 του αλγορίθμου):
        self.dq_da = self.critic_net.compute_dq_da(self.current_state_batch, action_for_gradient)

        # Update του actor (Βήμα 18 του αλγορίθμου):
        self.actor_net.train_actor(self.current_state_batch, self.dq_da)

        # Τελικό update του target actor & critic
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()


















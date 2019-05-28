import random
from collections import deque
import numpy as np

capacity = 10000
num_of_actions = 1


class ReplayMemory(object):

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.replay_memory = deque()
        self.position = 0
        self.time_step = 0

        # Συνάρτηση που βάζει ένα experience (s,a,r,s') στο RM
    def add_experience(self, current_state, next_state, action, reward):
        self.current_state = current_state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.replay_memory.append((self.current_state, self.next_state, self.action, self.reward))
        self.time_step += 1
        if len(self.replay_memory) > capacity:
            self.replay_memory.popleft()

        return self.replay_memory


    def minibatches(self, batch_size):  # Επιλέγει ένα τυχαίο batch από την μνήμη μεγέθους batch_size

        if len(self.replay_memory) < batch_size:
            batch = [[[0, 0], [0, 0], 0, 0] for i in range(batch_size)]
        else:
            batch = random.sample(self.replay_memory, batch_size)

        # State y(t)
        self.current_state_batch = [item[0] for item in batch]
        self.current_state_batch = np.array(self.current_state_batch)
        # print(self.current_state_batch)
        # Next State y(t+1)
        self.next_state_batch = batch[1]
        self.next_state_batch = np.array(self.next_state_batch)
        # Reward r(t)
        self.reward_batch = [item[2] for item in batch]
        self.reward_batch = np.array(self.reward_batch)

        # Action π(t)
        self.action_batch = [item[3] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch, [len(self.action_batch), num_of_actions])
        # print(self.next_state_batch)
        # return [self.current_state_batch, self.next_state_batch, self.action_batch, self.reward_batch]







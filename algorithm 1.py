import numpy as np
import random
import tensorflow as tf
from collections import deque


tf.enable_eager_execution()

# Hyper parameters
T = 200  # Time steps
gamma = tf.get_variable('γ', shape=[1], initializer=tf.constant_initializer[0.99])   # Discount factor
alpha = tf.get_variable('a', shape=[1], initializer=tf.constant_initializer[0.1])  # Learning Rate
epsilon = 0.001  # Reward function tolerance
batch = 20  # Batch size
tau = tf.get_variable('τ', shape=[1], initializer=tf.constant_initializer[0.01])  # Target update rate
rm = 100000  # Replay memory size
num_of_states = 2  # Ένα tuple (y(t),y_set)
num_of_actions = 1
c = tf.get_variable('reward_constant', shape=[1], initializer=tf.constant_initializer[5])

Q_table = np.zeros(num_of_states, num_of_actions)   # Q_table initialization
action = tf.placeholder(np.argmax(Q_table))   # Greedy Policy initialization
W_a = tf.get_variable('W_a', shape=[1], initializer=tf.constant_initializer[random.uniform(-1, 1)])  # Weight init.
W_c = tf.get_variable('W_c', shape=[1], initializer=tf.constant_initializer[random.uniform(-1, 1)])  # Weight init.
y_1 = 5.3    # Η συνάρτηση από την G(s) όταν t=1
y_set = tf.constant(tf.random.uniform(0, 1))    # Ένα τυχαίο set point
initial_state_tuple = (y_1, y_set)
initial_state = tf.placeholder(initial_state_tuple)

y = []

# Replay Memory
class ReplayMemory(object):

    def __init__(self, capacity=100000):   # Αρχικοποιεί την κλάση
        self.capacity = capacity   # Πόσο χωράει η μνήμη, ορίζεται από εμένα (εδώ 100000)
        self.memory = deque()          # Tι έχει μπει στην μνήμη (αρχίζει ως άδεια)
        self.position = 0            # Η αρχική θέση ενός entry της μνήμης

    def save(self, s, a, s_next, r):             # Σώζει μια επανάληψη
        experience = (s, a, s_next, r)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)      # Βάζει ενα state στην άδεια λίστα της μνήμης
            self.memory += 1
        else:                                # Αν ξεπεράσουμε την χωρητικότητα της μνήμης διαγράφουμε παλιά στοιχεία
            del self.memory[0]
            self.memory.append(experience)

    def sample_use(self, batch_size):      # Επιλέγει ένα τυχαίο batch από την μνήμη μεγέθους batch_size
        batch = []

        if self.position < batch_size:
            batch = random.sample(self.memory, self.position)
        else:
            batch = random.sample(self.memory, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s_next_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, s_next_batch


for t in range(T):
    s = (y(t), y_set)
    action = action

    # Reward Function
    def reward(t, y_set, y):
        if True:
            for j in range(t):
                abs(y - y_set) < epsilon
                return c




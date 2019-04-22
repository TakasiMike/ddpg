import random
from collections import deque
import numpy as np


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






rm = ReplayMemory(capacity=20)
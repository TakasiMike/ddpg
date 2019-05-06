from RM import ReplayMemory
import numpy as np
Batch = 20
RM = ReplayMemory(100)


for i in range(120):
    cr = [np.random.randint(1, 100), np.random.randint(1, 100)]
    a = np.random.randint(1, 100)
    ne = np.random.randint(1, 100)
    r = np.random.randint(1, 100)
    exp = np.array(RM.add_experience(cr, a, ne, r))

# print(exp)
opop = RM.minibatches(40)
# print(opop)
# for i in range(Batch):
#     print(opop[i][0][1])

print(opop)
print(len(opop))

# batch = [[0, 0] for i in range(10)]
# print(batch)

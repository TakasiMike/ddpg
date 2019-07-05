import tensorflow as tf
import math

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

class solv_diff:

    def __init__(self):

        self.sess = tf.InteractiveSession()
        self.F = tf.placeholder("float", [1, 1])
        self.T_j = tf.placeholder("float", [1, 1])

    def equations(self, state, t):

        Ca, T = state
        self.d_Ca = (self.F / V) * (Ca_in - Ca) - 2 * k0 * math.exp(E / (R * T)) * (Ca ** 2)
        self.d_T = (self.F / V) * (T_in - T) + 2 * (minus_DH / (density * Cp)) * k0 * math.exp(E / (R * T)) * (Ca ** 2) - \
                (UA / (V * density * Cp)) * (T - self.T_j)
        self.differential = [self.d_Ca, self.d_T]
        return self.differential

    def evaluate_next_state(self, flow, temperature):
        return self.sess.run(self.differential, feed_dict={self.F: flow, self.T_j: temperature})





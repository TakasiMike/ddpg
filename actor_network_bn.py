import tensorflow as tf
import math
from batch_normalization import batch_norm
import numpy as np

LEARNING_RATE = 0.0005
TAU = 0.001
BATCH_SIZE = 128
N_HIDDEN_1 = 450  #400
N_HIDDEN_2 = 400   #300


class ActorNet_bn:


    def __init__(self, num_states, num_actions):

            self.sess = tf.InteractiveSession()

            # actor network model parameters:
            self.actor_state_in = tf.placeholder("float", [None, num_states])
            self.W1_a = tf.Variable(
                tf.random_uniform([num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)), name="W1_a")
            self.B1_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)), name="B1_a")
            self.W2_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)), name="W2_a")
            self.B2_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)), name="B2_a")
            self.W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2, num_actions], -0.003, 0.003), name="W3_a")
            self.B3_a = tf.Variable(tf.random_uniform([num_actions], -0.003, 0.003), name="B3_a")

            self.is_training = tf.placeholder(tf.bool, [])
            self.H1_t = tf.matmul(self.actor_state_in, self.W1_a)
            self.H1_a_bn = batch_norm(self.H1_t, N_HIDDEN_1, self.is_training, self.sess)
            self.H1_a = tf.nn.relu(self.H1_a_bn.bnorm) + self.B1_a

            self.H2_t = tf.matmul(self.H1_a, self.W2_a)
            self.H2_a_bn = batch_norm(self.H2_t, N_HIDDEN_2, self.is_training, self.sess)
            self.H2_a = tf.nn.relu(self.H2_a_bn.bnorm) + self.B2_a
            self.actor_model = tf.matmul(self.H2_a, self.W3_a) + self.B3_a


            # target actor network model parameters:
            self.t_actor_state_in = tf.placeholder("float", [None, num_states])
            self.t_W1_a = tf.Variable(
                tf.random_uniform([num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)), name="t_W1_a")
            self.t_B1_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)), name="t_B1_a")
            self.t_W2_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)), name="t_W2_a")
            self.t_B2_a = tf.Variable(
                tf.random_uniform([N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)), name="t_B2_a")
            self.t_W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2, num_actions], -0.003, 0.003), name="t_W3_a")
            self.t_B3_a = tf.Variable(tf.random_uniform([num_actions], -0.003, 0.003), name="t_B3_a")

            self.t_is_training = tf.placeholder(tf.bool, [])
            self.t_H1_t = tf.matmul(self.t_actor_state_in, self.t_W1_a)
            self.t_H1_a_bn = batch_norm(self.t_H1_t, N_HIDDEN_1, self.t_is_training, self.sess, self.H1_a_bn)
            self.t_H1_a = tf.nn.relu(self.t_H1_a_bn.bnorm) + self.t_B1_a

            self.t_H2_t = tf.matmul(self.t_H1_a, self.t_W2_a)
            self.t_H2_a_bn = batch_norm(self.t_H2_t, N_HIDDEN_2, self.t_is_training, self.sess, self.H2_a_bn)
            self.t_H2_a = tf.nn.relu(self.t_H2_a_bn.bnorm) + self.t_B2_a
            self.t_actor_model = tf.matmul(self.t_H2_a, self.t_W3_a) + self.t_B3_a

            # cost of actor network:
            self.q_gradient_input = tf.placeholder("float", [None,
                                                             num_actions])
            self.actor_parameters = [self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,
                                     self.H1_a_bn.scale, self.H1_a_bn.beta, self.H2_a_bn.scale, self.H2_a_bn.beta]
            self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters,
                                                     -self.q_gradient_input/BATCH_SIZE)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).apply_gradients(
                zip(self.parameters_gradients, self.actor_parameters))
            # initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())
            self.saver = tf.train.Saver()
            self.saver.save(self.sess, 'DDPG_MIMO', global_step=1000)



            self.sess.run([
                self.t_W1_a.assign(self.W1_a),
                self.t_B1_a.assign(self.B1_a),
                self.t_W2_a.assign(self.W2_a),
                self.t_B2_a.assign(self.B2_a),
                self.t_W3_a.assign(self.W3_a),
                self.t_B3_a.assign(self.B3_a)])

            self.update_target_actor_op = [
                self.t_W1_a.assign(TAU * self.W1_a + (1 - TAU) * self.t_W1_a),
                self.t_B1_a.assign(TAU * self.B1_a + (1 - TAU) * self.t_B1_a),
                self.t_W2_a.assign(TAU * self.W2_a + (1 - TAU) * self.t_W2_a),
                self.t_B2_a.assign(TAU * self.B2_a + (1 - TAU) * self.t_B2_a),
                self.t_W3_a.assign(TAU * self.W3_a + (1 - TAU) * self.t_W3_a),
                self.t_B3_a.assign(TAU * self.B3_a + (1 - TAU) * self.t_B3_a),
                self.t_H1_a_bn.updateTarget,
                self.t_H2_a_bn.updateTarget,
            ]

    def evaluate_actor(self, state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in: state_t, self.is_training: False})

    def evaluate_target_actor(self, state_t_1):
        return self.sess.run(self.t_actor_model,
                             feed_dict={self.t_actor_state_in: state_t_1, self.t_is_training: False})

    def train_actor(self, actor_state_in, q_gradient_input):
        self.sess.run([self.optimizer, self.H1_a_bn.train_mean, self.H1_a_bn.train_var, self.H2_a_bn.train_mean,
                       self.H2_a_bn.train_var, self.t_H1_a_bn.train_mean, self.t_H1_a_bn.train_var,
                       self.t_H2_a_bn.train_mean, self.t_H2_a_bn.train_var],
                      feed_dict={self.actor_state_in: actor_state_in, self.t_actor_state_in: actor_state_in,
                                 self.q_gradient_input: q_gradient_input, self.is_training: True,
                                 self.t_is_training: True})

    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)


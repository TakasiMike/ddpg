import numpy as np
import tensorflow as tf
import math

learning_rate = 0.0001
batch_size = 64
tau = 0.001


class CriticNet:

    def __init__(self, num_of_states, num_of_actions, W1_c, W2_c, W3_c, B1_c, B2_c, B3_c, t_W1_c, t_W2_c, t_W3_c, t_B1_c, t_B2_c, t_B3_c, critic_state_in, t_critic_state_in, critic_action_in, t_critic_action_in):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

        # Παράμετροι του critic

        self.W1_c = W1_c
        self.B1_c = B1_c
        self.W2_c = W2_c
        self.B2_c = B2_c
        self.W3_c = W3_c
        self.B3_c = B3_c
        self.critic_state_in = critic_state_in
        self.critic_action_in = critic_action_in
        self.critic_q_model = self.create_critic_net(num_of_states, num_of_actions)

        # Παράμετροι του target network

        self.t_W1_c = t_W1_c
        self.t_B1_c = t_B1_c
        self.t_W2_c = t_W2_c
        self.t_B2_c = t_B2_c
        self.t_W3_c = t_W3_c
        self.t_B3_c = t_B3_c
        self.t_critic_state_in = t_critic_state_in
        self.t_critic_action_in = t_critic_action_in
        self.t_critic_q_model = self.create_critic_net(num_of_states, num_of_actions)
        self.q_value_in = tf.placeholder('float', [None, 1])

        self.sess.run([
            self.t_W1_c.assign(self.W1_c),
            self.t_W2_c.assign(self.W2_c),
            self.t_W3_c.assign(self.W3_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_B2_c.assign(self.B2_c),
            self.t_B3_c.assign(self.B3_c)])

        self.update_target_critic_op = [
            self.t_W1_c.assign(tau*self.W1_c + (1-tau)*self.t_W1_c),
            self.t_W2_c.assign(tau*self.W2_c + (1-tau)*self.t_W2_c),
            self.t_W3_c.assign(tau*self.W3_c + (1-tau)*self.t_W3_c),
            self.t_B1_c.assign(tau*self.B1_c + (1-tau)*self.t_B1_c),
            self.t_B2_c.assign(tau*self.B2_c + (1-tau)*self.t_B2_c),
            self.t_B3_c.assign(tau*self.B3_c + (1-tau)*self.t_B3_c)]

    def create_critic_net(self, num_of_states=2, num_of_actions=1):
        num_hidden_1 = 30
        num_hidden_2 = 30
        critic_state_in = tf.placeholder('float', [None, num_of_states])
        critic_action_in = tf.placeholder('float', [None, num_of_actions])
        W1_c = tf.Variable(tf.random.uniform[num_of_states, num_hidden_1])
        W2_c = tf.Variable(tf.random.uniform[num_hidden_1, num_hidden_2])
        W2_action_c = tf.Variable(tf.random.uniform[num_of_actions, num_hidden_2])
        W3_c = tf.Variable(tf.random.uniform[num_hidden_2, num_of_actions])
        B1_c = tf.Variable(tf.random.uniform[num_hidden_1])
        B2_c = tf.Variable(tf.random.uniform[num_hidden_2])
        B3_c = tf.Variable(tf.random.uniform[num_of_actions])

        # Forward Feed
        H1_c = tf.nn.sigmoid(tf.add(tf.matmul(critic_state_in, W1_c), B1_c))
        H2_c = tf.nn.sigmoid(tf.add(tf.matmul(H1_c, W2_c), B2_c))
        critic_q_model = tf.add(tf.matmul(H2_c, W3_c), B3_c)
        return W1_c, W2_c, W3_c, B1_c, B2_c, B3_c, critic_state_in, critic_action_in, critic_q_model, W2_action_c


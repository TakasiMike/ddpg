import numpy as np
import tensorflow as tf
import math

learning_rate = 0.0001
batch_size = 64
tau = 0.001


class ActorNet:

    def __init__(self, num_of_states, num_of_actions, W1_a, W2_a, W3_a, B1_a, B2_a, B3_a, t_W1_a, t_W2_a, t_W3_a, t_B1_a, t_B2_a, t_B3_a, actor_state_in, t_actor_state_in):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

        # Παράμετροι του actor

        self.W1_a = W1_a
        self.B1_a = B1_a
        self.W2_a = W2_a
        self.B2_a = B2_a
        self.W3_a = W3_a
        self.B3_a = B3_a
        self.actor_state_in = actor_state_in
        self.actor_model = self.create_actor_net(num_of_states, num_of_actions)

        # Παράμετροι του target network

        self.t_W1_a = t_W1_a
        self.t_B1_a = t_B1_a
        self.t_W2_a = t_W2_a
        self.t_B2_a = t_B2_a
        self.t_W3_a = t_W3_a
        self.t_B3_a = t_B3_a
        self.t_actor_state_in = t_actor_state_in
        self.t_actor_model = self.create_actor_net(num_of_states, num_of_actions)

        self.sess.run([
            self.t_W1_a.assign(self.W1_a),
            self.t_W2_a.assign(self.W2_a),
            self.t_W3_a.assign(self.W3_a),
            self.t_B1_a.assign(self.B1_a),
            self.t_B2_a.assign(self.B2_a),
            self.t_B3_a.assign(self.B3_a)])

        self.update_target_actor_op = [
            self.t_W1_a.assign(tau*self.W1_a + (1-tau)*self.t_W1_a),
            self.t_W2_a.assign(tau*self.W2_a + (1-tau)*self.t_W2_a),
            self.t_W3_a.assign(tau*self.W3_a + (1-tau)*self.t_W3_a),
            self.t_B1_a.assign(tau*self.B1_a + (1-tau)*self.t_B1_a),
            self.t_B2_a.assign(tau*self.B2_a + (1-tau)*self.t_B2_a),
            self.t_B3_a.assign(tau*self.B3_a + (1-tau)*self.t_B3_a)]

    def create_actor_net(self, num_of_states=2, num_of_actions=1):
        num_hidden_1 = 30
        num_hidden_2 = 30
        actor_state_in = tf.placeholder('float', [None, num_of_states])
        W1_a = tf.Variable(tf.random.uniform[num_of_states, num_hidden_1])
        W2_a = tf.Variable(tf.random.uniform[num_hidden_1, num_hidden_2])
        W3_a = tf.Variable(tf.random.uniform[num_hidden_2, num_of_actions])
        B1_a = tf.Variable(tf.random.uniform[num_hidden_1])
        B2_a = tf.Variable(tf.random.uniform[num_hidden_2])
        B3_a = tf.Variable(tf.random.uniform[num_of_actions])

        # Forward Feed
        H1_a = tf.nn.sigmoid(tf.add(tf.matmul(actor_state_in, W1_a), B1_a))
        H2_a = tf.nn.sigmoid(tf.add(tf.matmul(H1_a, W2_a), B2_a))
        actor_model = tf.add(tf.matmul(H2_a, W3_a), B3_a)
        return W1_a, W2_a, W3_a, B1_a, B2_a, B3_a, actor_state_in, actor_model












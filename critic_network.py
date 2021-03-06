import tensorflow as tf
import numpy as np

learning_rate = 0.01
tau = 0.001


class CriticNet:

    def __init__(self, num_of_states, num_of_actions):
        # self.g = tf.Graph()
        # with self.g.as_default():
        #     self.sess = tf.Session()

        self.sess = tf.Session()
        # Παράμετροι του critic network

        self.W1_c, self.B1_c, self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c, \
            self.critic_state_in, \
            self.critic_action_in, self.critic_q_model,  = self.create_critic_net(num_of_states, num_of_actions)

        # Παράμετροι του target critic network
        self.t_W1_c, self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c, \
            self.t_critic_state_in, \
            self.t_critic_action_in, self.t_critic_q_model = self.create_critic_net(num_of_states, num_of_actions)



        # Σχηματισμός της συνάρτησης κόστους του critic, η οποία θα ελαχιστοποιηθεί ως προς τα βάρη Wc
        self.q_value_in = tf.placeholder("float", [None, 1])
        self.l2_regularizer_loss = 0.1 * tf.reduce_sum(tf.pow(self.W2_c, 2)) + 0.1 * tf.reduce_sum(tf.pow(self.B2_c, 2))
        self.cost = tf.reduce_sum(tf.pow(self.q_value_in - self.critic_q_model, 2)) + self.l2_regularizer_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in, name='gradients',
                                       unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.action_gradients = [self.act_grad_v[0] / tf.to_float(tf.shape(self.act_grad_v[0])[0])]

        self.sess.run(tf.global_variables_initializer())

        self.sess.run([
            self.t_W1_c.assign(self.W1_c),
            self.t_W2_c.assign(self.W2_c),
            self.t_W3_c.assign(self.W3_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_B2_c.assign(self.B2_c),
            self.t_B3_c.assign(self.B3_c),
            self.t_W2_action_c.assign(self.W2_action_c)
        ])


        # Κανόνας update του target νευρωνικού δικτύου

        self.update_target_critic_op = [
            self.t_W1_c.assign(tau*self.W1_c + (1-tau)*self.t_W1_c),
            self.t_W2_c.assign(tau*self.W2_c + (1-tau)*self.t_W2_c),
            self.t_W3_c.assign(tau*self.W3_c + (1-tau)*self.t_W3_c),
            self.t_B1_c.assign(tau*self.B1_c + (1-tau)*self.t_B1_c),
            self.t_B2_c.assign(tau*self.B2_c + (1-tau)*self.t_B2_c),
            self.t_B3_c.assign(tau*self.B3_c + (1-tau)*self.t_B3_c),
            self.t_W2_action_c.assign(tau*self.W2_action_c + (1-tau)*self.t_W2_action_c)
        ]

        #  Action Gradient (dQ/da)
        # self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in, name='gradients', unconnected_gradients='zero')



    # Δημιουργία του νευρωνικού δικτύου του critic. Τελικά παράγεται το Q-table, δηλαδή το critic_q_model

    def create_critic_net(self, num_of_states=2, num_of_actions=1):
        num_hidden_1 = 300
        num_hidden_2 = 300
        critic_state_in = tf.placeholder('float', [None, num_of_states], name="state_in")
        critic_action_in = tf.placeholder('float', [None, num_of_actions], name="action_in")
        W1_c = tf.Variable(tf.random.uniform([num_of_states, num_hidden_1]))
        W2_c = tf.Variable(tf.random.uniform([num_hidden_1, num_hidden_2]))
        W2_action_c = tf.Variable(tf.random.uniform([num_of_actions, num_hidden_2]))
        W3_c = tf.Variable(tf.random.uniform([num_hidden_2, num_of_actions]))
        B1_c = tf.Variable(tf.random.uniform([num_hidden_1]))
        B2_c = tf.Variable(tf.random.uniform([num_hidden_2]))
        B3_c = tf.Variable(tf.random.uniform([num_of_actions]))

        # Forward Feed
        H1_c = tf.nn.sigmoid(tf.add(tf.matmul(critic_state_in, W1_c), B1_c), name="H1C")
        H2_c = tf.nn.sigmoid(tf.add(tf.matmul(H1_c, W2_c), B2_c), name="H2C")
        critic_q_model = tf.matmul(H2_c, W3_c) + B3_c
        return W1_c, B1_c, W2_c, W2_action_c,  B2_c, W3_c, B3_c, critic_state_in, critic_action_in, critic_q_model

    # Συνάρτηση που εκπαιδεύει το critic network
    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch,
                                                 self.critic_action_in: action_batch, self.q_value_in: y_i_batch})

    # Δίνουμε ως input το s' και α' και παίρνουμε το target critic network

    def evaluate_target_network(self, next_state_batch, next_action_batch):
        return self.sess.run(self.t_critic_q_model,
                             feed_dict={self.t_critic_state_in: next_state_batch, self.t_critic_action_in: next_action_batch})

    # Συνάρτηση που υπολογίζει το dQ/dα. Σαν inputs έχει το s και α

    def compute_dq_da(self, state_t, action_t):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t})
    #
    # def compute_dq_dw(self, state_t, weights_t):
    #     return self.sess.run(self.weight_grad_v,
    #                          feed_dict={self.critic_state_in: state_t, self.critic_parameters: weights_t})

    # Συνάρτηση που κάνει update το critic target network

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)





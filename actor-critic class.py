
import tensorflow as tf


class ActorCritic(object):   # Κλάση που θα περιέχει τους actor & critic
    def __init__(self, act_dim, st_dim):
        self.act_dim = act_dim
        self.st_dim = st_dim
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, [None, st_dim], 's')
        self.s_next = tf.placeholder(tf.float32, [None, st_dim], 's_next')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')


# Συνάρτηση που χτίζει το δίκτυο του actor

     def build_actor(self, state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=None, custom_getter=custom_getter):
            n_layer_1 = 20
            w_a = tf.get_variable('w_a', [self.act_dim, n_layer_1], trainable=trainable)
            bias_act = tf.get_variable('bias_act', [1, n_layer_1], trainable=trainable)
            net_act = tf.nn.sigmoid(tf.add(tf.matmul(state, w_a), bias_act))
            policy = tf.layers.dense(net_act, 1, trainable=trainable)
            return policy  # π(s,Wa)



            #net = tf.layers.dense(state, 20, activation=tf.nn.sigmoid, trainable=trainable)
            #action = tf.layers.dense(net, self.act_dim, activation=tf.nn.sigmoid)
            #return action  # π(s)

# Συνάρτηση που χτίζει τον critic

     def build_c(self, state, action, reuse=None, custom_getter=None):
        trainable = True if reuse in None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            nodes_layer_1 = 20
            w_c = tf.get_variable('w1_s', [self.st_dim + self.act_dim, nodes_layer_1], trainable=trainable)
            #w1_a = tf.get_variable('w1_a', [self.act_dim, nodes_layer_1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, nodes_layer_1], trainable=trainable)
            net = tf.nn.sigmoid(tf.add(tf.matmul(state, w_c), tf.matmul(action, w_c), b1))
            q = tf.layers.dense(net, 1, trainable=trainable)
            return q   # Q(s,a,Wc)















import tensorflow as tf
tf.enable_eager_execution()




class ActorCritic(object):   # Κλάση που θα περιέχει τους actor & critic
    w_a = tf.Variable()
    w_c = tf.Variable()
    def __init__(self, act_dim, st_dim):
        self.act_dim = act_dim
        self.st_dim = st_dim
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, [None, st_dim], 's')
        self.s_next = tf.placeholder(tf.float32, [None, st_dim], 's_next')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = self.build_actor(self.s)
        self.q = self.build_c(self.s, self.a)
        # self.w_a = w_a
        # self.w_c = w_c
#

# Συνάρτηση που χτίζει το δίκτυο του actor

    def build_actor(self, state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=None, custom_getter=custom_getter):
            n_layer_1 = 20
            self.w_a = tf.global_variables('w_a', [self.act_dim, n_layer_1], trainable=trainable)
            bias_act = tf.get_variable('bias_act', [1, n_layer_1], trainable=trainable)
            net_act = tf.nn.sigmoid(tf.add(tf.matmul(state, self.w_a),  bias_act))
            policy = tf.layers.dense(net_act, 1, trainable=trainable)
            return policy  # π(s,Wa)

    # def test(self):
    #     self.a

            #net = tf.layers.dense(state, 20, activation=tf.nn.sigmoid, trainable=trainable)
            #action = tf.layers.dense(net, self.act_dim, activation=tf.nn.sigmoid)
            #return action  # π(s)

# Συνάρτηση που χτίζει τον critic

    def build_c(self, state, action, reuse=None, custom_getter=None):
        trainable = True if reuse in None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            nodes_layer_1 = 20
            self.w_c = tf.global_variables('w1_s', [self.st_dim + self.act_dim, nodes_layer_1], trainable=trainable)
            # w1_a = tf.get_variable('w1_a', [self.act_dim, nodes_layer_1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, nodes_layer_1], trainable=trainable)
            net = tf.nn.sigmoid(tf.add(tf.matmul(state, self.w_c), tf.matmul(action, self.w_c), b1))
            q = tf.layers.dense(net, 1, trainable=trainable)
            return q   # Q(s,a,Wc)


# Μερική παράγωγος του π ως προς Wa

    def partial_a(self):
        with tf.GradientTape() as tape:
            pol = self.build_actor(self.s)   # Εδώ θα μπει σαν input το state του αντιστοιχου time step
            dp_da = tape.gradient(pol, self.w_a)
            return dp_da

# Μερική παράγωγος του Q ως προς α

    def partial_q_a(self):
        with tf.GradientTape() as tape:
            q_table = self.build_c(self.s, self.a)
            pol = self.build_actor(self.s)  # Εδώ θα μπει σαν input το state του αντιστοιχου time step
            dq_da = tape.gradient(q_table, pol)
            return dq_da

# Μερική παράγωγος του Q ως προς w_c

    def partial_q_wc(self):
        with tf.GradientTape() as tape:
            q_table = self.build_c(self.s, self.a)
            dq_dw = tape.gradient(q_table, self.w_c)
            return dq_dw

# Υπολογισμός reward function

    def reward(self):























import tensorflow as tf


class grad_inverter:

    def __init__(self, action_bounds):

        self.action_size = 1
        self.sess = tf.InteractiveSession()
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size], name="action")
        self.p_max = tf.constant(action_bounds[0], dtype=tf.float32)
        self.p_min = tf.constant(action_bounds[1], dtype=tf.float32)
        self.p_range = self.p_max - self.p_min
        self.p_diff_max = tf.div(-self.action_input + self.p_max, self.p_range)
        self.p_diff_min = tf.div(self.action_input - self.p_min, self.p_range)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])  # Το ανάδελτα ως προς p
        self.grad_inverter = tf.where(tf.math.greater(self.act_grad, self.zeros_act_grad_filter),
                                      tf.math.multiply(self.act_grad, self.p_diff_max),
                                      tf.math.multiply(self.act_grad, self.p_diff_min))

    def inverter(self, grad, action):

        return self.sess.run(self.grad_inverter, feed_dict={self.act_grad: grad[0], self.action_input: action})



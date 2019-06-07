import tensorflow as tf


class Grad_Inverter:

    def __init__(self, action_bounds):

        self.action_size = len(action_bounds[0])  # 1
        self.sess = tf.InteractiveSession()
        self.action_input = tf.placeholder(tf.float32, [None, 1])
        self.p_max = tf.constant(action_bounds[0], dtype=tf.float32)
        self.p_min = tf.constant(action_bounds[1], dtype=tf.float32)
        self.p_range = tf.constant([x - y for x, y in zip(action_bounds[0], action_bounds[1])], dtype=tf.float32)
        self.p_diff_max = tf.div(-self.action_input + self.p_max, self.p_range)
        self.p_diff_min = tf.div(self.action_input - self.p_min, self.p_range)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])  # Το ανάδελτα ως προς p
        self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter),
                                      tf.math.multiply(self.act_grad, self.p_diff_max),
                                      tf.math.multiply(self.act_grad, self.p_diff_min))

    def inverter(self, grad, action):

        return self.sess.run(self.grad_inverter, feed_dict={self.action_input: action, self.act_grad: grad[0]})



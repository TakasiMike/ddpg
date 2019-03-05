import gym
import numpy as np
import tensorflow as tf


class ActorCritic(object):   # Κλάση που θα περιέχει τους actor & critic
    def __init__(self, act_dim, st_dim):
        self.act_dim
        self.st_dim
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, [None, st_dim], 's')
        self.s_next = tf.placeholder(tf.float32, [None, st_dim], 's_next')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')













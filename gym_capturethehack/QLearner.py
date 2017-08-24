import tensorflow as tf
import numpy as np
from gym_capturethehack.config import config

class QLearner:
    def __init__(self, number_of_team_members):
        """ init the model with hyper-parameters etc """
        self.previous_q = None
        self.previous_action = None
        self.y = 0.8 # Discount factor
        self.e = 0.1 # Epsilon for e-greedy choice
        
        move = np.linspace(-1.0, 1.0, 21)
        rotation = np.linspace(-1, 1, 21)
        shoot = [0, 1]
        communicate = [0, 1]
        self.actions = [(m, r, s, c) for m in move for r in rotation for s in shoot for c in communicate]

        self.number_of_team_members = number_of_team_members

        self.img = tf.placeholder(tf.float32, shape=(1,)+config["image_size"], name="image")
        self.comm = tf.placeholder(tf.float32, shape=(1, self.number_of_team_members), name="comm")
        self.conv1 = tf.layers.conv2d(self.img, filters=8, kernel_size=8, strides=4, activation=tf.nn.elu, name="conv1")
        self.conv2 = tf.layers.conv2d(self.conv1, filters=16, kernel_size=4, strides=3, activation=tf.nn.elu, name="conv2")
        self.flat = tf.contrib.layers.flatten(self.conv2)
        self.flatcomm = tf.concat([self.flat, self.comm], axis=1)
        self.fc1 = tf.layers.dense(self.flatcomm, units=1200, activation=tf.nn.elu, name="fc1")
        self.fc2 = tf.layers.dense(self.fc1, units=1000, activation=tf.nn.elu, name="fc2")
        self.out = tf.layers.dense(self.fc2, units=len(self.actions), name="out")
        self.predict = tf.argmax(self.out, axis=1)

        self.next_q = tf.placeholder(tf.float32, shape=(1, len(self.actions)))
        self.loss = tf.reduce_sum(tf.squared_difference(self.next_q, self.out))

        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()

    def inference(self, img, comm):
        o, p = self.sess.run([self.out, self.predict], feed_dict={self.img: img, self.comm: comm})
        self.previous_q = o

        action = self.actions[p[0]]
        if np.random.rand() < self.e:
            action = np.random.choice(self.actions)
        self.previous_action = action

        return (o, action)

    def optimize(self, img, comm, target_Q):
        _, l = self.sess.run([self.train, self.loss], feed_dict={self.next_q: target_Q, self.img: img, self.comm: comm})
        return l

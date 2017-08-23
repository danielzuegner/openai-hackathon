import tensorflow as tf
import numpy as np
from config import config

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
        self.actions = [(m, r, s) for m in move for r in rotation for s in shoot]

        self.output_dim = output_dim
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

        self.sess = tf.Session()

    def inference(self, img, comm):
        o, p = self.sess.run([self.out, self.predict], feed_dict={self.img: img, self.comm: comm})
        self.previous_q = o
        self.previous_action = self.actions[p[0]] # TODO: Choose action in a epsilon greedy manner
        return o, self.actions[p[0]]


    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x)
        self.loss = tf.loss_function(y, y_predict, name="loss") # supervised
        # loss = tf.loss_function(x, y_predicted) # unsupervised

    def optimize(self, batch_x, batch_y):
        return tf.train.AdamOptimizer(0.001).minimize(self.loss, name="optimizer")

import tensorflow as tf
import numpy as np
from gym_capturethehack.config import config

class QLearner:
    def __init__(self, number_of_team_members, id, team):

        """ init the model with hyper-parameters etc """
        self.y = 0.8 # Discount factor
        self.e = 0.1 # Epsilon for e-greedy choice
        self.id = id
        self.team = team
        
        move = np.linspace(-1.0, 1.0, 21)
        rotation = np.linspace(-1, 1, 21)
        shoot = [0, 1]
        communicate = [0, 1]
        self.actions = [(m, r, s, c) for m in move for r in rotation for s in shoot for c in communicate]

        self.previous_q = [0 for _ in self.actions]
        self.previous_action = 0
        self.number_of_team_members = number_of_team_members

        self.img = tf.placeholder(tf.float32, shape=(1,)+config["image_size"], name="{}-{}_image".format(id, team))
        self.conv1 = tf.layers.conv2d(self.img, filters=8, kernel_size=8, strides=4, activation=tf.nn.elu, name="{}-{}_conv1".format(id,team))
        self.conv2 = tf.layers.conv2d(self.conv1, filters=16, kernel_size=4, strides=3, activation=tf.nn.elu, name="{}-{}_conv2".format(id,team))
        self.flat = tf.contrib.layers.flatten(self.conv2)
        self.fc1 = tf.layers.dense(self.flat, units=1200, activation=tf.nn.elu, name="{}-{}_fc1".format(id,team))
        self.fc2 = tf.layers.dense(self.fc1, units=1000, activation=tf.nn.elu, name="{}-{}_fc2".format(id,team))
        self.out = tf.layers.dense(self.fc2, units=len(self.actions), name="{}-{}_out".format(id,team))
        self.predict = tf.argmax(self.out, axis=1)

        self.next_q = tf.placeholder(tf.float32, shape=(1, len(self.actions)))
        self.loss = tf.reduce_sum(tf.squared_difference(self.next_q, self.out))

        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def inference(self, img):
        o, p = self.sess.run([self.out, self.predict], feed_dict={self.img: img})
        self.previous_q = o

        action = self.actions[p[0]]
        choice = p[0]
        ixs = len(self.actions)
        if np.random.rand() < self.e:
            choice = np.random.choice(ixs)
            action = self.actions[choice]
        self.previous_action = choice

        return (o, action)

    def optimize(self, img, target_Q):
        _, l = self.sess.run([self.train, self.loss], feed_dict={self.next_q: np.expand_dims(np.squeeze(target_Q),0), self.img: img/255})
        return l

    def print_statistics(self):
        all_weights = np.array([v.eval(self.sess).reshape(-1) for v in tf.trainable_variables()
                                if '{}-{}'.format(self.id, self.team) in v.name])
        w = []
        for weight in all_weights:
            w.extend(weight.tolist())

        all_weights = np.array(w)

        print('**********************************')
        print('Weight Statistics for Agent {}, Team {}'.format(self.id, self.team))
        print("Mean: {}\n Std.: {}\n Min: {}\n Max: {}".format(np.mean(all_weights), np.std(all_weights),
                                                               np.min(all_weights), np.max(all_weights)))
        print('**********************************')

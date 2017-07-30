import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import pandas as pd
import statistics as st
import math

class kbo_wrcp:

    def __init__(self, lr):
        self.rate_learning = lr

    def data_preprocess(self, file, file2):
        df = pd.read_csv(file, '\t')
        self.X = np.array(df.values[:, :27])
        for i in range(self.X.shape[1]):
            mean, stdv = st.mean(self.X[:, i]), st.stdev(self.X[:, i])
            if (stdv != 0):
                for j in range(self.X.shape[0]):
                    self.X[j, i] = (self.X[j, i] - mean) / stdv
        df = pd.read_csv(file2, '\t')
        self.Y = np.array(df.values[:, :])
        for i in range(self.Y.shape[1]):
            mean, stdv = st.mean(self.Y[:, i]), st.stdev(self.Y[:, i])
            if (stdv != 0):
                for j in range(self.Y.shape[0]):
                    self.Y[j, i] = (self.Y[j, i] - mean) / stdv

    def data_training(self):
        input = tf.placeholder("float", [self.X.shape[0], self.X.shape[1]])
        answer = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])
        coeff = tf.Variable(tf.random_normal([self.X.shape[1], 1]))
        bias = tf.Variable(tf.random_normal([1]))
        output = tf.nn.sigmoid(tf.add(tf.matmul(input, coeff), bias))
        cost = tf.reduce_mean(tf.pow(output - answer, 2))
        opti = tf.train.GradientDescentOptimizer(self.rate_learning).minimize(cost)
        result = tf.transpose(coeff)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5000001):
           c, _, r= sess.run([cost, opti, result], feed_dict={input: self.X, answer: self.Y})
           if (iter % 1000000 == 0):
               print(iter,  "cost: ", c, "coeffs are")
               print(r)

test = kbo_wrcp(0.01)
test.data_preprocess('kbodata.tsv', 'kbowrcp.tsv')
test.data_training()
a = 0

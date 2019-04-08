import sys
sys.path.append("..")
import tensorflow as tf
from RNNCell import GRUDCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import time
from sklearn import metrics
import numpy as np
import os
from datautils import helper
from .RNNModel import RNNModel

class GRU(RNNModel):
    '''
    Class to run the GRUD model as described in https://www.nature.com/articles/s41598-018-24271-9
    '''
    def __init__(self,
                sess,
                args,
                train_data,
                validation_data,
                test_data):
        super(GRU, self).__init__(sess,
                                    args,
                                    train_data,
                                    validation_data,
                                    test_data)

    def RNN(self, x, m, delta, mean, x_lengths):
        with tf.variable_scope('GRU', reuse=tf.AUTO_REUSE):
            # shape of x = [batch_size, n_steps, n_inputs]
            # shape of m = [batch_size, n_steps, n_inputs]
            # shape of delta = [batch_size, n_steps, n_inputs]
            X = tf.reshape(x, [-1, self.n_inputs])
            M = tf.reshape(m, [-1, self.n_inputs])
            Delta = tf.reshape(delta, [-1, self.n_inputs])
            X = tf.reshape(X, [-1, self.n_steps, self.n_inputs])

            grud_cell = tf.nn.rnn_cell.GRUCell(num_units=self.n_hidden_units,
                                                activation=None, # Uses tanh if None
                                                reuse=tf.AUTO_REUSE,
                                                kernel_initializer=self.kernel_initializer,#Orthogonal initializer
                                                bias_initializer=self.bias_initializer)
            
            grud_cell=tf.nn.rnn_cell.DropoutWrapper(grud_cell,output_keep_prob=self.keep_prob)
            init_state = grud_cell.zero_state(self.batch_size, dtype=tf.float32) # Initializing first hidden state to zeros
            outputs, _ = tf.nn.dynamic_rnn(grud_cell, X, 
                                            initial_state=init_state,
                                            sequence_length=x_lengths,
                                            time_major=False)
            
            outputs=tf.reshape(outputs,[-1, self.n_hidden_units])
            outputs = tf.nn.dropout(outputs,keep_prob=self.keep_prob)
            outputs = tf.layers.dense(outputs,units=self.n_hidden_units,
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.dense(outputs,units=self.n_classes, 
                                        kernel_initializer=self.kernel_initializer)
            outputs=tf.reshape(outputs,[-1,self.n_steps,self.n_classes])
            
        return outputs

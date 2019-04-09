import sys
sys.path.append("..")
import tensorflow as tf
from .RNNModel import RNNModel

class LSTMSimple(RNNModel):
    def __init__(self, sess, args, train_data, validation_data,test_data):
        super(LSTMSimple, self).__init__(sess,args,train_data,validation_data,
                                    test_data)

    def RNN(self, x, m, delta, mean, x_lengths):
        with tf.variable_scope('GRUD', reuse=tf.AUTO_REUSE):
            # shape of x = [batch_size, n_steps, n_inputs]
            # shape of m = [batch_size, n_steps, n_inputs]
            # shape of delta = [batch_size, n_steps, n_inputs]
            X = tf.reshape(x, [-1, self.n_inputs])
            M = tf.reshape(m, [-1, self.n_inputs])
            Delta = tf.reshape(delta, [-1, self.n_inputs])
            X = tf.concat([X,M,Delta], axis=1)
            X = tf.reshape(X, [-1, self.n_steps, 3*self.n_inputs])

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden_units,
                                               activation=None,
                                               # Uses tanh if None
                                               reuse=tf.AUTO_REUSE)

            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=self.keep_prob)
            init_state = lstm_cell.zero_state(self.batch_size,
                                              dtype=tf.float32)  # Initializing first hidden state to zeros
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, X,
                                           initial_state=init_state,
                                           sequence_length=x_lengths,
                                           time_major=False)

            outputs = tf.reshape(outputs, [-1, self.n_hidden_units])
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            outputs = tf.layers.dense(outputs, units=self.n_hidden_units,
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                          1e-4))
            outputs = tf.nn.relu(outputs)
            outputs = tf.layers.dense(outputs, units=self.n_classes,
                                      kernel_initializer=self.kernel_initializer)
            outputs = tf.reshape(outputs, [-1, self.n_steps, self.n_classes])

        return outputs


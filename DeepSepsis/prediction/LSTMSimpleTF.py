"""
Author: Srinivasan Sivanandan
"""
import sys

sys.path.append("..")
import tensorflow as tf
# from RNNCell import LSTMSimpleCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.layers import dense
import time
from sklearn import metrics
import numpy as np


class LSTMSimpleTF():
    '''
    Class to run the GRUD model as described in https://www.nature.com/articles/s41598-018-24271-9
    '''

    def __init__(self,
                 sess,
                 args,
                 train_data,
                 test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.result_path = args.result_path
        self.epochs = args.epochs
        self.n_inputs = args.n_inputs
        self.n_hidden_units = args.n_hidden_units
        self.n_classes = args.n_classes
        self.checkpoint_dir = args.checkpoint_dir
        self.normalize = args.normalize
        self.log_dir = args.log_dir
        self.dropout_rate = args.dropout_rate
        self.n_steps = train_data.maxLength
        self.threshold = args.threshold
        self.experiment = args.experiment

        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32,
                                [None, self.n_steps, self.n_classes])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.delta = tf.placeholder(tf.float32,
                                    [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs, ])
        self.x_lengths = tf.placeholder(tf.float32, [self.batch_size, ])
        self.sess = sess

    def getModelDir(self, epoch):
        return "{}_{}_{}_{}/epoch{}".format(self.experiment, self.lr,
                                            self.batch_size, self.normalize,
                                            epoch)

    def RNN(self, x, m, delta, mean, x_lengths):
        with tf.variable_scope('LSTMSimple', reuse=tf.AUTO_REUSE):
            # shape of x = [batch_size, n_steps, n_inputs]
            # shape of m = [batch_size, n_steps, n_inputs]
            # shape of delta = [batch_size, n_steps, n_inputs]
            X = tf.reshape(x, [-1, self.n_inputs])

            # print("X shape: " + str(x.shape))

            # ---------Is this needed for LSTM? ------------
            M = tf.reshape(m, [-1, self.n_inputs])
            Delta = tf.reshape(delta, [-1, self.n_inputs])
            X = tf.concat([X, M, Delta], axis=1)
            X = tf.reshape(X, [-1, self.n_steps, 3 * self.n_inputs])
            # ---------------------------------------------

            # lstm_simple_cell = LSTMSimpleCell.LSTMSimpleCell(input_size=self.n_inputs,
            #                               hidden_size=self.n_hidden_units,
            #                               indicator_size=self.n_inputs,
            #                               delta_size=self.n_inputs,
            #                               output_size=1,
            #                               dropout_rate=self.dropout_rate,
            #                               xMean=mean,
            #                               activation=None,  # Uses tanh if None
            #                               reuse=tf.AUTO_REUSE,
            #                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #                               # MSRA initializer
            #                               bias_initializer=tf.initializers.ones(),
            #                               # ones - commonly used in LSTM http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            #                               name=None,
            #                               dtype=None)

            """
            __init__(
            num_units,
            use_peepholes=False,
            cell_clip=None,
            initializer=None,
            num_proj=None,
            proj_clip=None,
            num_unit_shards=None,
            num_proj_shards=None,
            forget_bias=1.0,
            state_is_tuple=True,
            activation=None,
            reuse=None,
            name=None,
            dtype=None,
            **kwargs
            )
            
            """

            lstm_simple_cell = LSTMCell(self.n_inputs)

            init_state = lstm_simple_cell.zero_state(self.batch_size,
                                              dtype=tf.float32)  # Initializing first hidden state to zeros
            outputs, _ = tf.nn.dynamic_rnn(lstm_simple_cell, X, \
                                           initial_state=init_state, \
                                           sequence_length=x_lengths,
                                           time_major=False)
            outputs = dense(outputs, 1, activation=tf.nn.sigmoid)
        return outputs

    def build(self):
        self.pred = self.RNN(self.x, self.m, self.delta, self.mean,
                             self.x_lengths)

        positive_class = tf.reduce_sum(tf.cast((self.y == 1), dtype=tf.float32))
        negative_class = tf.reduce_sum(tf.cast((self.y == 0), dtype=tf.float32))
        class_ratio = negative_class / (positive_class + 1)
        # class_ratio=30 - Change class ratio - left for experimentation

        print(self.y.shape)
        print(self.pred.shape)

        self.loss = tf.reduce_sum(
            tf.nn.weighted_cross_entropy_with_logits(targets=self.y,
                                                     logits=self.pred,
                                                     pos_weight=class_ratio))
        # self.cross_entropy = -tf.reduce_sum(self.y*tf.log(self.pred)) # RNN return logits
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.y_pred = tf.cast((self.pred > self.threshold), dtype=tf.int32)
        self.accuracy, self.update_op = tf.metrics.accuracy(labels=self.y,
                                                            predictions=self.y_pred)
        self.saver = tf.train.Saver(max_to_keep=None)
        loss_sum = tf.summary.scalar("loss", self.loss)
        acc_sum = tf.summary.scalar("acc", self.accuracy)
        self.sum = tf.summary.merge([loss_sum, acc_sum])
        self.board = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def load_model(self, epoch):
        import re
        import os
        checkpoint_dir = os.path.join(self.checkpoint_dir,
                                      self.getModelDir(epoch), self.experiment)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer(r"(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Checkpoint not found")
            return False, 0

    def save_model(self, epoch, step):
        import os
        checkpoint_dir = os.path.join(self.checkpoint_dir,
                                      self.getModelDir(epoch))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir,
                                                self.experiment + '.model'),
                        global_step=step)

    def train(self):
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        start_time = time.time()
        idx = 0
        epochcount = 0
        dataset = self.train_data
        counter = 0
        while epochcount < self.epochs:
            dataset.shuffle()
            for train_x, train_y, train_m, train_delta, train_xlen in dataset.getNextBatch():
                _, loss, summary_str, acc, _ = self.sess.run(
                    [self.train_op, self.loss, self.sum, self.accuracy,
                     self.update_op], feed_dict={ \
                        self.x: train_x,
                        self.y: train_y,
                        self.m: train_m,
                        self.delta: train_delta,
                        self.x_lengths: train_xlen,
                        self.mean: dataset.mean,
                    })
                self.board.add_summary(summary_str, counter)
                counter += 1
            epochcount += 1

            self.save_model(counter, epochcount)
            acc, auc = self.test()
            print("epoch is : %2.2f, Accuracy: %.8f, AUC: %.8f" % (
            epochcount, acc, auc))

        return auc

    def test(self):
        start_time = time.time()
        counter = 0
        dataset = self.test_data
        dataset.shuffle()
        target = []
        predictions = []
        for test_x, test_y, test_m, test_delta, test_xlen in dataset.getNextBatch():
            summary_str, acc, pred = self.sess.run(
                [self.sum, self.accuracy, self.pred], feed_dict={
                    self.x: test_x,
                    self.y: test_y,
                    self.m: test_m,
                    self.delta: test_delta,
                    self.mean: dataset.mean,
                    self.x_lengths: test_xlen,
                })
            # Remove padding for accuracy and AUC calculation
            for i in range(0, test_xlen.shape[0]):
                target.extend(list(test_y[i, 0:test_xlen[i]]))
                predictions.extend(list(pred[i, 0:test_xlen[i]]))

        auc = metrics.roc_auc_score(np.array(target), np.array(predictions))
        predictions = np.array(np.array(predictions) > self.threshold).astype(
            int)
        acc = metrics.accuracy_score(np.array(target), np.array(predictions))
        # Also compute utility score
        return acc, auc


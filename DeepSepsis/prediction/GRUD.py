"""
Author: Srinivasan Sivanandan
"""
import sys
sys.path.append("..")
import tensorflow as tf
from RNNCell import GRUDCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import time
from sklearn import metrics
import numpy as np

class grud():
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
        self.y = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs,])
        self.x_lengths = tf.placeholder(tf.float32, [self.batch_size,])
        self.y_mask = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.utp = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.ufp = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.ufn = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # Output Weights
        self.kernel_initializer = tf.initializers.orthogonal()
        self.bias_initializer = tf.initializers.ones()
        # self.output_kernel = tf.get_variable(
        #                             "output/weights" ,
        #                             shape=[self.n_hidden_units, self.n_classes],
        #                             initializer=self.kernel_initializer,
        #                             trainable=True)
        # self.output_bias = tf.get_variable(
        #                             "output/bias",
        #                             shape=[self.n_classes],
        #                             initializer=self.bias_initializer,
        #                             trainable=True)

        self.sess = sess

    def getModelDir(self, epoch):
        return "{}_{}_{}_{}/epoch{}".format(self.experiment, self.lr,
                                            self.batch_size, self.normalize, epoch)

    def RNN(self, x, m, delta, mean, x_lengths):
        with tf.variable_scope('GRUD', reuse=tf.AUTO_REUSE):
            # shape of x = [batch_size, n_steps, n_inputs]
            # shape of m = [batch_size, n_steps, n_inputs]
            # shape of delta = [batch_size, n_steps, n_inputs]
            X = tf.reshape(x, [-1, self.n_inputs])
            M = tf.reshape(m, [-1, self.n_inputs])
            Delta = tf.reshape(delta, [-1, self.n_inputs])
            #X = tf.concat([X,M,Delta], axis=1)
            X = tf.concat([X,M], axis=1)
            #X = tf.reshape(X, [-1, self.n_steps, 3*self.n_inputs])
            X = tf.reshape(X, [-1, self.n_steps, 2*self.n_inputs])

            # grud_cell = GRUDCell.GRUDCell(input_size=self.n_inputs,
            #                             hidden_size=self.n_hidden_units,
            #                             indicator_size=self.n_inputs,
            #                             delta_size=self.n_inputs,
            #                             output_size = 1,
            #                             dropout_rate = self.dropout_rate,
            #                             xMean = mean, 
            #                             activation=None, # Uses tanh if None
            #                             reuse=tf.AUTO_REUSE,
            #                             kernel_initializer=self.kernel_initializer,#Orthogonal initializer
            #                             bias_initializer=self.bias_initializer,#ones - commonly used in LSTM http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            #                             name=None,
            #                             dtype=None)

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
                                        activation='relu', 
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            outputs = tf.layers.dense(outputs,units=self.n_classes, 
                                        kernel_initializer=self.kernel_initializer)
            outputs=tf.reshape(outputs,[-1,self.n_steps,self.n_classes])

        return outputs

    def build(self):
        self.pred = self.RNN(self.x, self.m, self.delta, self.mean, self.x_lengths)

        positive_class = tf.reduce_sum(tf.cast((self.y>0.5), dtype=tf.float32))
        padding = tf.reduce_sum(tf.cast((self.y_mask<0.5), dtype=tf.float32))
        negative_class = tf.reduce_sum(tf.cast((self.y<0.5), dtype=tf.float32))-padding

        self.class_ratio = (1.0*negative_class)/((0.05*negative_class))  #- Change class ratio - left for experimentation
        
        self.act = tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.pred, pos_weight=self.class_ratio)
        self.act = self.y_mask*self.act
        
        self.y_act = tf.sigmoid(self.pred)
        self.utility = (self.y*self.y_act)*self.utp + (self.y*(1.0-self.y_act))*self.ufn + ((1.0-self.y)*self.y_act)*self.ufp
        self.utility= self.y_mask*self.utility
        self.utility = tf.reduce_mean(self.utility)

        self.loss = tf.reduce_mean(self.act)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        self.y_pred = tf.cast((self.pred>self.threshold),dtype=tf.int32)
        self.accuracy, self.acc_update = tf.metrics.accuracy(labels=self.y,predictions=self.y_pred)
        self.tp, self.tp_update = tf.metrics.true_positives(labels=self.y,predictions=self.y_pred)
        self.tpr = (1.0*self.tp)/(positive_class+1)
        self.fp, self.fp_update = tf.metrics.false_positives(labels=self.y,predictions=self.y_pred)
        self.fpr = (1.0*self.fp)/(negative_class+1)

        self.metric_op = tf.group(self.acc_update, self.accuracy, self.tp_update, self.tp, self.fp_update, self.fp, self. tpr, self.fpr, self.class_ratio)

        self.saver = tf.train.Saver(max_to_keep=None)
        loss_sum = tf.summary.scalar("loss", self.loss)
        acc_sum = tf.summary.scalar("acc", self.accuracy)
        tp_sum = tf.summary.scalar("TP", self.tp)
        tpr_sum = tf.summary.scalar("TPR", self.tpr)
        fpr_sum = tf.summary.scalar("FPR", self.fpr)
        fp_sum = tf.summary.scalar("FP", self.fp)
        utility_sum = tf.summary.scalar("Utility", self.utility)

        self.sum=tf.summary.merge([loss_sum, acc_sum, tp_sum, tpr_sum, fpr_sum, fp_sum, utility_sum])
        self.board = tf.summary.FileWriter(self.log_dir,self.sess.graph)
    
    def load_model(self, epoch):
        import re
        import os
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.getModelDir(epoch), self.experiment)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer(r"(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Checkpoint not found")
            return False, 0
    
    def save_model(self,epoch,step):
        import os
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.getModelDir(epoch))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.experiment+'.model'), global_step=step)


    def train(self):
        tf.global_variables_initializer().run()
        start_time=time.time()
        idx = 0
        epochcount=0
        dataset=self.train_data
        counter = 0
        while epochcount<self.epochs:
            tf.local_variables_initializer().run()
            dataset.shuffle()
            for train_x,train_y,train_m,train_delta,train_xlen,y_mask,utp,ufp,ufn in dataset.getNextBatch():
                _,loss,summary_str,acc,_, cr = self.sess.run([self.train_op,self.loss, self.sum, self.accuracy, self.metric_op, self.class_ratio], feed_dict={
                    self.x: train_x,
                    self.y: train_y,
                    self.m: train_m,
                    self.delta: train_delta, 
                    self.x_lengths: train_xlen,                   
                    self.mean: dataset.mean,
                    self.y_mask:y_mask,
                    self.utp:utp,
                    self.ufp:ufp,
                    self.ufn:ufn,
                    self.keep_prob:self.dropout_rate
                })
                counter += 1
            self.board.add_summary(summary_str, epochcount)
            epochcount+=1
            
            self.save_model(counter, epochcount)
            acc,auc=self.test()
            print("epoch is : %2.2f, Accuracy: %.8f, AUC: %.8f, Class Ratio: %.8f" % (epochcount, acc, auc, cr))

        return auc

    def test(self):
        start_time=time.time()
        counter=0
        dataset = self.test_data
        dataset.shuffle()
        target = []
        predictions = []
        for test_x,test_y,test_m,test_delta,test_xlen,y_mask,utp,ufp,ufn  in dataset.getNextBatch():
            summary_str,acc,pred = self.sess.run([self.sum, self.accuracy,self.pred], feed_dict={
                self.x: test_x,
                self.y: test_y,
                self.m: test_m,
                self.delta: test_delta,
                self.mean: dataset.mean,
                self.x_lengths: test_xlen,
                self.y_mask:y_mask,
                self.utp:utp,
                self.ufp:ufp,
                self.ufn:ufn,
                self.keep_prob:1.0
            })
            # Remove padding for accuracy and AUC calculation
            for i in range(0,test_xlen.shape[0]):
                target.extend(list(test_y[i, 0:test_xlen[i]]))
                predictions.extend(list(pred[i, 0:test_xlen[i]]))

        auc = metrics.roc_auc_score(np.array(target),np.array(predictions))
        predictions = np.array(np.array(predictions)>self.threshold).astype(int)
        acc = metrics.accuracy_score(np.array(target),np.array(predictions))
        # Also compute utility score
        return acc, auc

    
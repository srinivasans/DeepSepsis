"""
Author: Srinivasan Sivanandan
"""
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import time
from sklearn import metrics
import numpy as np
import os

'''
_,loss,summary_str,acc,_, cr,out = sess.run([model.train_op, model.loss, model.sum, model.accuracy, model.metric_op, model.class_ratio, model.output], feed_dict={
                    model.x: train_x,
                    model.y: train_y,
                    model.m: train_m,
                    model.delta: train_delta, 
                    model.x_lengths: train_xlen,                   
                    model.mean: dataset.mean,
                    model.y_mask:y_mask,
                    model.utp:utp,
                    model.ufp:ufp,
                    model.ufn:ufn,
                    model.keep_prob:model.dropout_rate,
                    model.isTrain:True
                })
'''

class DAE():
    '''
    Class to run the GRUD model as described in https://www.nature.com/articles/s41598-018-24271-9
    '''
    def __init__(self,
                sess,
                args,
                train_data,
                val_data,
                test_data):
        self.train_data = train_data
        self.val_data = val_data
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
        self.early_stopping_patience = args.early_stopping_patience
        self.seed = args.seed

        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs,])
        self.x_lengths = tf.placeholder(tf.float32, [self.batch_size,])
        self.y_mask = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.utp = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.ufp = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.ufn = tf.placeholder(tf.float32, [None, self.n_steps, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.isTrain = tf.placeholder(tf.bool)

        # Output Weights
        self.kernel_initializer = tf.initializers.glorot_uniform(seed=self.seed)
        self.bias_initializer = tf.initializers.zeros()

        self.sess = sess

    def getModelDir(self, epoch):
        return "{}_{}_{}_{}/epoch{}".format(self.experiment, self.lr,
                                            self.batch_size, self.normalize, epoch)

    def RNN(self, x, m, delta, mean, x_lengths):
        with tf.variable_scope('DAE', reuse=tf.AUTO_REUSE):
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
            outputs = tf.layers.dense(outputs,units=self.n_inputs, 
                                        kernel_initializer=self.kernel_initializer)
            outputs=tf.reshape(outputs,[-1,self.n_steps,self.n_inputs])
            
        return outputs

    def build(self):
        self.pred = self.RNN(self.x, self.m, self.delta, self.mean, self.x_lengths)
        self.output = self.pred
        
        self.loss = tf.reduce_sum(tf.squared_difference(self.pred*self.m,self.y*self.m))/tf.reduce_sum(self.m)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
 
        self.saver = tf.train.Saver(max_to_keep=None)
        loss_sum = tf.summary.scalar("loss", self.loss)
        
        self.sum=tf.summary.merge([loss_sum])
        self.board = tf.summary.FileWriter(self.log_dir,self.sess.graph)
    
    def load_model(self, epoch, checkpoint_dir=None):
        import re
        import os

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.checkpoint_dir, self.getModelDir(epoch))

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
        min_mse = np.inf
        early_stopping_counter = 0
        while epochcount<self.epochs:
            tf.local_variables_initializer().run()
            dataset.shuffle()
            for train_x,train_y,train_m,train_delta,train_xlen,y_mask,utp,ufp,ufn,files,labels in dataset.getNextBatch(epoch=epochcount):
                _,loss,summary_str = self.sess.run([self.train_op, self.loss, self.sum], feed_dict={
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
                    self.keep_prob:self.dropout_rate,
                    self.isTrain:True
                })
                counter += 1
                self.board.add_summary(summary_str, counter)
            epochcount+=1
            
            if epochcount%1==0:
                self.save_model(epochcount, epochcount)

            test_counter = (epochcount-1)*counter/epochcount
            val_loss=self.test(val=True,counter=test_counter)
            print("epoch is : %2.2f, TrainLoss(MSE): %.8f, ValLoss(MSE): %.8f" % (epochcount, loss, val_loss))

            # Early Stopping
            if(val_loss < min_mse):
                min_mse = val_loss
                early_stopping_counter = 0
                best_epoch = epochcount
            else:
                early_stopping_counter+=1
            
            # if early_stopping_counter >= self.early_stopping_patience:
            #     print("Early Stopping Training : Max MSE = %f , Best Epoch : %d"%(min_mse, best_epoch))
            #     break

        return min_mse, best_epoch



    def save_output(self,predictions,labels,filenames):
        for i in range(0,len(predictions)):
            folder = os.path.join(self.result_path,self.experiment)
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, filenames[i])
            with open(filename,'w') as f:
                f.write('HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel\n')
                # For each time step
                for j in range(0, len(predictions[i])):
                    # For each feature
                    for k in range(0,len(predictions[i][j])):
                        f.write(str(predictions[i][j][k])+'|')
                    f.write('0|0|0|'+str(j)+'|'+str(labels[i][j][0])+'\n')

    def test(self, val=True, checkpoint_dir=None, test_epoch=100, generate_files=False,counter=0, load_checkpoint=False):
        start_time=time.time()

        if val:
            dataset = self.val_data
        else:
            dataset = self.test_data

        dataset.shuffle()
        target = []
        predictions = []
        test_files = []
        predictions_ind = []
        labels_ind = []

        if load_checkpoint:
            self.load_model(test_epoch, checkpoint_dir)

        tf.local_variables_initializer().run()

        squared_error = 0.0
        num_samples = 0
        for test_x,test_y,test_m,test_delta,test_xlen,y_mask,utp,ufp,ufn,files,test_labels  in dataset.getNextBatch(epoch=30):
            summary_str,pred,val_loss = self.sess.run([self.sum, self.output, self.loss], feed_dict={
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
                self.keep_prob:1.0,
                self.isTrain:False
            })
            # Remove padding for accuracy and AUC calculation
            pred = (pred*(1.0-test_delta))+(test_y*test_delta)
            squared_error+= np.sum(((pred-test_y)*test_m*y_mask)**2)
            num_samples+=np.sum(test_m) - np.sum(test_delta)

            # Reset the non-missing values to actual known values 
            pred = (pred*(1.0-test_m))+(test_y*test_m)
            for i in range(0,test_xlen.shape[0]):
                outputs = pred[i, 0:test_xlen[i]]*dataset.std+dataset.mean
                predictions_ind.append(list(outputs))
                labels_ind.append(list(test_labels[i, 0:test_xlen[i]]))
                test_files.append(files[i])

                
            if generate_files:
                self.save_output(predictions_ind,labels_ind,test_files)

            mse = squared_error/num_samples
        return mse

    
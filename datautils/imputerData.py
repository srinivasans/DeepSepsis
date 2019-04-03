'''
Class to read data and yield batched data for training
Author: Srinivasan Sivanandan
'''
import os
import numpy as np
import random
import data

class ImputerData(data.Data):

    def __init__(self, path, files, batchSize = 100, isTrain=True, normalize=True, padding=True, mean = None, std = None, maxLength=0.0):
        super(ImputerData, self).__init__(path,files,batchSize,isTrain,normalize,padding,mean,std,maxLength)

    def getNextBatch(self, epoch=0, minMaskEpoch=10):
        cursor = 0
        while cursor+self.batchSize <= self.x.shape[0]:
            x = self.x[cursor:cursor+self.batchSize]
            
            # Simulate missingness only during Training
            if self.isTrain and epoch > minMaskEpoch:
                mask = np.random.choice([0, 1], size=self.batchSize*self.x.shape[1]*self.x.shape[2], p=[0.4, 0.6])
                mask = mask.reshape(x.shape)
                x = x*mask
            
            y = self.x[cursor:cursor+self.batchSize]
            labels = self.y[cursor:cursor+self.batchSize]
            y_mask = self.y_mask[cursor:cursor+self.batchSize]
            m = self.m[cursor:cursor+self.batchSize]
            d = self.timeDelay[cursor:cursor+self.batchSize]
            xlen = self.x_lengths[cursor:cursor+self.batchSize]
            utp = self.UTP[cursor:cursor+self.batchSize]
            ufp = self.UFP[cursor:cursor+self.batchSize]
            ufn = self.UFN[cursor:cursor+self.batchSize]
            files = self.files[cursor:cursor+self.batchSize]

            cursor+=self.batchSize
            yield x,y,m,d,xlen,y_mask,utp,ufp,ufn,files,labels
    


'''
Class to read data and yield batched data for training
Author: Srinivasan Sivanandan
'''
import os
import numpy as np
import random

class ReadData():

    def __init__(self, path, batchSize = 100, isTrain=True, normalize=True, padding=True, mean = None, std = None, maxLength=0.0):
        # Phase parameters
        self.path=path
        self.isTrain=isTrain
        self.normalize = normalize
        self.padding = padding
        self.batchSize = batchSize

        # Sepsis specific parameters
        self.features = {'HR':0, 'O2Sat':1, 'Temp':2,
                    'SBP':3,'MAP':4, 'DBP':5,
                    'Resp':6, 'EtCO2':7, 'BaseExcess':8,
                    'HCO3':9, 'FiO2':10, 'pH':11, 'PaCO2':12,
                    'SaO2':13, 'AST':14, 'BUN':15, 'Alkalinephos':16,
                    'Calcium':17, 'Chloride':18, 'Creatinine':19,
                    'Bilirubin_direct':20, 'Glucose':21, 'Lactate':22, 
                    'Magnesium':23, 'Phosphate':24, 'Potassium':25, 'Bilirubin_total':26,
                    'TroponinI':27, 'Hct':28, 'Hgb':29, 'PTT':30, 
                    'WBC':31, 'Fibrinogen':32, 'Platelets':33, 'Age':34, 
                    'Gender':35, 'ICULOS':39, 'SepsisLabel':40}
        # All features except ICULOS and SepsisLabel
        self.nX =  len(self.features)-2
        self.nY = 1 # Sepsis Label
        
        # Data variables
        self.x = []
        self.y = []
        self.m = []
        self.timeDelay = []
        self.times = []
        self.files = []
        self.x_lengths = []
        self.y_mask = []
        self.UTP = []
        self.UFN = []
        self.UFP = []

        # Aggregate variables
        self.maxLength = maxLength # Count for padding using mini-batches

        # Read data (Mean and std passed on to load!)
        self.load(mean=mean, std=std) 

    def readFile(self, input_file):
        with open(os.path.join(self.path,input_file), 'r') as f:
            header = f.readline().strip()
            column_names = header.split('|')
            values = np.loadtxt(f, delimiter='|')

        target = None
        times = None
        if column_names[-1] == 'SepsisLabel':
            target = values[:,-1]
            column_names = column_names[:-1]
            values = values[:, :-1]
        
        if column_names[-1] == 'ICULOS':
            times = values[:,-1]
            column_names = column_names[:-1]
            values = values[:, :-1]

        values = values[:,0:self.nX]
        delay = np.zeros(values.shape)
        for j in range(0, values.shape[1]):
            for i in range(1, values.shape[0]):
                if(np.isnan(values[i,j])):
                    delay[i] = times[i]-times[i-1]+delay[i-1,j]
                else:
                    delay[i] = times[i]-times[i-1]

        indicator = np.array(~np.isnan(values)).astype(int)
        return values, target, indicator, times, delay


    def load(self, mean=None, std=None):
        input_files = os.listdir(self.path)
        
        self.x = []
        self.y = []
        self.m = []
        self.timeDelay = []
        self.times = []
        self.files = []
        self.x_lengths = []
        for input_file in input_files:
            x,y,m,t,d = self.readFile(input_file)
            self.x.append(x)
            self.y.append(y)
            self.m.append(m)
            self.times.append(t)
            self.timeDelay.append(d)
            self.files.append(input_file)
            self.x_lengths.append(x.shape[0])

            if self.isTrain and x.shape[0] > self.maxLength:
                self.maxLength = x.shape[0]
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.m = np.array(self.m)
        self.times = np.array(self.times)
        self.timeDelay = np.array(self.timeDelay)

        if self.padding:
            x_values = self.x
            m_values = self.m
            delta_values = self.timeDelay
            y_values = self.y
            self.x = np.full([x_values.shape[0], self.maxLength, self.nX], np.nan)
            self.m = np.full([m_values.shape[0], self.maxLength, self.nX], np.nan)
            self.timeDelay = np.full([delta_values.shape[0], self.maxLength, self.nX], np.nan)
            self.y = np.full([y_values.shape[0], self.maxLength], np.nan)
            self.y_mask = np.ones([y_values.shape[0], self.maxLength])
            self.UTP = np.zeros([y_values.shape[0], self.maxLength])
            self.UFN = np.zeros([y_values.shape[0], self.maxLength])
            self.UFP = np.zeros([y_values.shape[0], self.maxLength])
            
            for i in range(0, x_values.shape[0]):
                assert x_values[i].shape[1]==self.nX
                self.x[i,0:x_values[i].shape[0],:] = x_values[i][:,:]
                self.m[i,0:m_values[i].shape[0],:] = m_values[i][:,:]
                self.timeDelay[i,0:delta_values[i].shape[0],:] = delta_values[i][:,:]
                self.y[i,0:y_values[i].shape[0]] = y_values[i]

                # Padded y
                #self.y[i,y_values[i].shape[0]:] = y_values[i][y_values[i].shape[0]-1]
                self.y_mask[i,y_values[i].shape[0]:] = 0
                
                # Calculate Utility functions
                self.UFP[i,:] = -0.025
                pos = -1
                for k in range(0,y_values[i].shape[0]):
                    if self.y[i,k]>0.5:
                        pos=k
                        break
                
                if pos>=0:
                    self.y[i,pos-6:pos]=1 # Fill -12->-6
                    self.UFN[i,pos:pos+9]=np.array([-2.0*p/9.0 for p in range(0,np.min([9,self.maxLength-pos]))])
                    self.UFN[i,pos+9:]=-2.0
                    #self.UTP[i,0:pos-8]=-0.05 # Not required already taken care by FP
                    if pos<7:
                        self.UTP[i,0:pos]=np.array([(7-pos+p)/7.0 for p in range(0,pos)])
                    else:   
                        self.UTP[i,pos-6:pos+1]=np.array([p/7.0 for p in range(0,7)])
                    self.UTP[i,pos:pos+9]=np.array([1-(p/9.0) for p in range(0,np.min([9,self.maxLength-pos]))])

            self.y = np.reshape(self.y,(y_values.shape[0], self.maxLength,1))
            self.y_mask = np.reshape(self.y_mask,(y_values.shape[0], self.maxLength,1))
            self.UTP = np.reshape(self.UTP,(y_values.shape[0], self.maxLength,1))
            self.UFN = np.reshape(self.UFN,(y_values.shape[0], self.maxLength,1))
            self.UFP = np.reshape(self.UFP,(y_values.shape[0], self.maxLength,1))

        if self.isTrain:
            x_values = np.reshape(self.x, [-1, self.nX])
            self.mean = np.nanmean(x_values, axis=0)
            self.std = np.nanstd(x_values, axis=0)
        else:
            self.mean = mean
            self.std = std
        
        if self.normalize:
            self.x = np.nan_to_num((self.x - self.mean) / self.std)
            self.m = np.nan_to_num(self.m)
            self.timeDelay = np.nan_to_num(self.timeDelay)
            self.y = np.nan_to_num(self.y)
        
        
    def getMean(self):
        if self.normalize:
            return np.array([0.0]*self.nX)
        else:
            return self.mean
        
    def shuffle(self):
        c = list(zip(self.x,self.y,self.m,self.timeDelay,self.times, self.x_lengths))
        random.shuffle(c)
        self.x,self.y,self.m,self.timeDelay,self.times, self.x_lengths=zip(*c)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.m = np.array(self.m)
        self.timeDelay = np.array(self.timeDelay)
        self.times = np.array(self.times)
        self.x_lengths = np.array(self.x_lengths)
        
    def getNextBatch(self):
        cursor = 0
        while cursor+self.batchSize <= self.x.shape[0]:
            x = self.x[cursor:cursor+self.batchSize]
            y = self.y[cursor:cursor+self.batchSize]
            y_mask = self.y_mask[cursor:cursor+self.batchSize]
            m = self.m[cursor:cursor+self.batchSize]
            d = self.timeDelay[cursor:cursor+self.batchSize]
            xlen = self.x_lengths[cursor:cursor+self.batchSize]
            utp = self.UTP[cursor:cursor+self.batchSize]
            ufp = self.UFP[cursor:cursor+self.batchSize]
            ufn = self.UFN[cursor:cursor+self.batchSize]

            cursor+=self.batchSize
            yield x,y,m,d,xlen,y_mask,utp,ufp,ufn
    


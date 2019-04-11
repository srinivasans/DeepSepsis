'''
Class to read data and yield batched data for training
Author: Srinivasan Sivanandan
'''
import os
import numpy as np
import random
from tqdm import tqdm

class Data():

    def __init__(self, path, files, batchSize = 100, 
                isTrain=True, normalize=True, padding=True, 
                mean = None, std = None, maxLength=0, 
                imputeForward=False, calculateDelay=True, imputation_folder=None):
                
        # Phase parameters
        self.path=path
        self.files = files
        self.isTrain=isTrain
        self.normalize = normalize
        self.padding = padding
        self.batchSize = batchSize
        self.imputeForward = imputeForward
        self.calculateDelay = calculateDelay
        self.imputationFolder = imputation_folder

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
        self.x_lengths = []
        self.y_mask = []
        self.UTP = []
        self.UFN = []
        self.UFP = []

        self.input_dim = self.nX
        self.output_dim = 1

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

        if self.calculateDelay:
            for j in range(0, values.shape[1]):
                for i in range(1, values.shape[0]):
                    if(np.isnan(values[i,j])):
                        delay[i,j] = times[i]-times[i-1]+delay[i-1,j]
                    else:
                        delay[i,j] = times[i]-times[i-1]

        indicator = np.array(~np.isnan(values)).astype(int)
        return values, target, indicator, times, delay


    def load(self, mean=None, std=None):
        self.x = []
        self.y = []
        self.m = []
        self.timeDelay = []
        self.times = []
        self.x_lengths = []
        print("* Reading data...")
        for input_file in tqdm(self.files):
            x,y,m,t,d = self.readFile(input_file)

            if self.imputationFolder is not None:
                self.path = self.imputationFolder
                x,y_,m_,t_,d_ = self.readFile(input_file)
                self.path = self.imputationFolder
            
            self.x.append(x)
            self.y.append(y)
            self.m.append(m)
            self.times.append(t)
            self.timeDelay.append(d)
            self.x_lengths.append(x.shape[0])

            if self.isTrain and x.shape[0] > self.maxLength:
                self.maxLength = x.shape[0]
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.m = np.array(self.m)
        self.times = np.array(self.times)
        self.timeDelay = np.array(self.timeDelay)

        x_values = self.x
        m_values = self.m
        delta_values = self.timeDelay
        y_values = self.y
        times_values = self.times
        self.x = np.full([x_values.shape[0], self.maxLength, self.nX], np.nan)
        self.m = np.full([m_values.shape[0], self.maxLength, self.nX], np.nan)
        self.timeDelay = np.full([delta_values.shape[0], self.maxLength, self.nX], np.nan)
        self.y = np.full([y_values.shape[0], self.maxLength], np.nan)
        self.y_mask = np.ones([y_values.shape[0], self.maxLength])
        self.UTP = np.zeros([y_values.shape[0], self.maxLength])
        self.UFN = np.zeros([y_values.shape[0], self.maxLength])
        self.UFP = np.zeros([y_values.shape[0], self.maxLength])
        self.times = np.full([times_values.shape[0], self.maxLength], np.nan)
        
        print("* Processing data...")
        for i in tqdm(range(0, x_values.shape[0])):
            assert x_values[i].shape[1]==self.nX
            self.x[i,0:x_values[i].shape[0],:] = x_values[i][:,:]
            self.m[i,0:m_values[i].shape[0],:] = m_values[i][:,:]
            self.timeDelay[i,0:delta_values[i].shape[0],:] = delta_values[i][:,:]
            self.y[i,0:y_values[i].shape[0]] = y_values[i]
            self.times[i,0:times_values[i].shape[0]] = times_values[i]

            # Create y-mask
            self.y_mask[i,y_values[i].shape[0]:] = 0
            
            # Calculate Utility functions
            self.UFP[i,:] = -0.025
            pos = -1
            for k in range(0,y_values[i].shape[0]):
                if self.y[i,k]>0.5:
                    pos=k
                    break
            
            if pos>=0:
                #self.y[i,pos-6:pos]=1 # Fill -12->-6
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
            # Do not normalize gender
            self.x[:,:,0:self.nX-1] = (self.x[:,:,0:self.nX-1] - self.mean[0:self.nX-1]) / self.std[0:self.nX-1]
        
        if self.imputeForward:
            # For each patient
            print("* Imputing Forward...")
            for i in tqdm(range(0, self.x.shape[0])):
                # For each feature
                for k in range(0, self.x.shape[2]):
                    # For each time step - from time 2
                    for j in range(1, self.x_lengths[i]):
                        if(np.isnan(self.x[i,j,k])):
                            self.x[i,j,k] = self.x[i,j-1,k]

        self.x = np.nan_to_num(self.x)
        self.m = np.nan_to_num(self.m)
        self.timeDelay = np.nan_to_num(self.timeDelay)
        self.y = np.nan_to_num(self.y)
        self.times = np.nan_to_num(self.times)

        # Remove padding if padding is False
        if not self.padding:
            print("* Unpadding...")
            num_patients = self.x.shape[0]
            x_values = self.x
            y_values = self.y
            m_values = self.m
            timeDelay_values = self.timeDelay
            y_mask_values = self.y_mask
            UTP_values = self.UTP
            UFN_values = self.UFN
            UFP_values = self.UFP
            times_values = self.times

            self.x = []
            self.m = []
            self.timeDelay = []
            self.y = []
            self.y_mask = []
            self.UTP = []
            self.UFN = []
            self.UFP = []
            self.times = []
            
            for i in tqdm(range(0,num_patients)):
                self.x.append(x_values[i,0:self.x_lengths[i],:])
                self.m.append(m_values[i,0:self.x_lengths[i],:])
                self.timeDelay.append(timeDelay_values[i,0:self.x_lengths[i],:]) 
                self.y.append(y_values[i,0:self.x_lengths[i],:])
                self.y_mask.append(y_mask_values[i,0:self.x_lengths[i],:]) 
                self.UTP.append(UTP_values[i,0:self.x_lengths[i],:])
                self.UFN.append(UFN_values[i,0:self.x_lengths[i],:])
                self.UFP.append(UFP_values[i,0:self.x_lengths[i],:])
                self.times.append(times_values[i,0:self.x_lengths[i]])

            self.x = np.array(self.x)
            self.m = np.array(self.m)
            self.timeDelay = np.array(self.timeDelay)
            self.y = np.array(self.y)
            self.y_mask = np.array(self.y_mask)
            self.UTP = np.array(self.UTP)
            self.UFP = np.array(self.UFP)
            self.UFN = np.array(self.UFN)
            self.times = np.array(self.times)
            
        
    def getMean(self):
        if self.normalize:
            return np.array([0.0]*self.nX)
        else:
            return self.mean
        
    def shuffle(self):
        c = list(zip(self.x,self.y,self.m,self.timeDelay,self.times, self.x_lengths,self.files, self.y_mask, self.UTP, self.UFP, self.UFN))
        random.shuffle(c)
        self.x,self.y,self.m,self.timeDelay,self.times, self.x_lengths,self.files, self.y_mask, self.UTP, self.UFP, self.UFN=zip(*c)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.m = np.array(self.m)
        self.y_mask = np.array(self.y_mask)
        self.timeDelay = np.array(self.timeDelay)
        self.times = np.array(self.times)
        self.x_lengths = np.array(self.x_lengths)
        self.files = np.array(self.files)
        self.UTP = np.array(self.UTP)
        self.UFP = np.array(self.UFP)
        self.UFN = np.array(self.UFN)
        
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
            files = self.files[cursor:cursor+self.batchSize]
            t = self.times[cursor:cursor+self.batchSize]
            t = np.expand_dims(t, axis=2)

            cursor+=self.batchSize
            yield x,y,m,d,xlen,y_mask,utp,ufp,ufn,files,t
    
    def getBatchGenerator(self, shuffle=False):
        while True:
            if shuffle:
                self.shuffle()
            gen = self.getNextBatch()
            val = next(gen, None)
            while val is not None:
                x,y,m,_,_,y_mask,_,_,_,_,t = val

                weights = y * 70.0 + (1-y) * 1.0
                weights = weights * y_mask
                weights = np.reshape(weights, (weights.shape[0],weights.shape[1]))

                val = next(gen, None)
                yield ([x,m,t], y, weights)

    def getInputGenerator(self):
        for inputs, _, _ in self.getBatchGenerator():
            yield inputs

    def getTargets(self):
        return self.y

    def getSteps(self):
        return self.x.shape[0] // self.batchSize

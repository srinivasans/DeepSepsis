'''
Class to generate and handle Train-Validation-Test split of data
Author: Srinivasan Sivanandan
'''
import os
import numpy as np
from .imputerData import ImputerData

class Dataset():

    def __init__(self, path, batchSize = 100, train_ratio=0.8, normalize=True, padding=True, seed=42, missingRate=0.4, minMaskEpoch=10):
        self.path = path
        self.batchSize = batchSize
        self.normalize = normalize
        self.padding = padding
        self.input_files = np.array(os.listdir(self.path))
        self.seed = seed
        self.missingRate = missingRate
        self.minMaskEpoch = minMaskEpoch
        
        self.dataset_size = len(self.input_files)
        self.train_size = int(np.round(self.dataset_size*train_ratio))
        self.val_size = int(np.round(self.dataset_size*(1.0-train_ratio)/2.0))
        self.test_size = self.dataset_size - self.train_size - self.val_size

        np.random.seed(42)
        np.random.shuffle(self.input_files)
        self.test_files = self.input_files[self.train_size+self.val_size:]
        
        self.train_val_files = self.input_files[0:self.train_size+self.val_size]
        np.random.seed(self.seed)
        np.random.shuffle(self.train_val_files)
        self.train_files = self.train_val_files[0:self.train_size]
        self.val_files = self.train_val_files[self.train_size:self.train_size+self.val_size]
        assert len(self.test_files)==self.test_size
        
        #Max length across all datasets = 336. 
        #Setting min maxLength=336 for traindata for now!!
        #TODO: Find max of max lengths across all datasets and use that for setting this maxLength
        self.train_data = ImputerData(path, 
                                    files=self.train_files, 
                                    batchSize = self.batchSize, 
                                    isTrain=True, 
                                    normalize=self.normalize, 
                                    padding=self.padding, 
                                    mean = None, 
                                    std = None, 
                                    maxLength=336,
                                    missingRate=self.missingRate,
                                    minMaskEpoch=self.minMaskEpoch,
                                    induceMissingness=True)
        
        self.val_data = ImputerData(path,
                                    files=self.val_files,
                                    batchSize=self.batchSize,
                                    isTrain=False,
                                    normalize=self.normalize,
                                    padding=self.padding,
                                    mean=self.train_data.mean,
                                    std=self.train_data.std,
                                    maxLength=self.train_data.maxLength,
                                    missingRate=self.missingRate,
                                    minMaskEpoch=self.minMaskEpoch,
                                    induceMissingness=True)

        self.test_data = ImputerData(path,
                                    files=self.test_files,
                                    batchSize=self.batchSize,
                                    isTrain=False,
                                    normalize=self.normalize,
                                    padding=self.padding,
                                    mean=self.train_data.mean,
                                    std=self.train_data.std,
                                    maxLength=self.train_data.maxLength,
                                    missingRate=self.missingRate,
                                    minMaskEpoch=self.minMaskEpoch,
                                    induceMissingness=False)

    
        


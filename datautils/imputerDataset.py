'''
Class to generate and handle Train-Validation-Test split of data
Author: Srinivasan Sivanandan
'''
import os
import numpy as np
import imputerData

class ImputerDataset():

    def __init__(self, path, batchSize = 100, train_ratio=0.8, normalize=True, padding=True):
        self.path = path
        self.batchSize = batchSize
        self.normalize = normalize
        self.padding = padding
        self.input_files = np.array(os.listdir(self.path))
        
        np.random.seed(42)
        
        np.random.shuffle(self.input_files)
        self.dataset_size = len(self.input_files.shape)
        self.train_size = int(self.dataset_size*train_ratio)
        self.val_size = int(self.dataset_size*(1.0-train_ratio)/2.0)
        self.test_size = int(self.dataset_size*(1.0-train_ratio)/2.0)

        self.train_files = self.input_files[0:self.train_size]
        self.val_files = self.input_files[self.train_size:self.train_size+self.val_size]
        self.test_files = self.input_files[self.train_size+self.val_size:]
        assert len(self.test_files)==self.test_size
        
        self.train_data = imputerData.ImputerData(path, 
                                    files=self.train_files, 
                                    batchSize = self.batchSize, 
                                    isTrain=True, 
                                    normalize=self.normalize, 
                                    padding=self.padding, 
                                    mean = None, 
                                    std = None, 
                                    maxLength=0.0)
        
        self.val_data = imputerData.ImputerData(path,
                                    files=self.val_files,
                                    batchSize=self.batchSize,
                                    isTrain=False,
                                    normalize=self.normalize,
                                    padding=self.padding,
                                    mean=self.train_data.mean,
                                    std=self.train_data.std,
                                    maxLength=self.train_data.maxLength)

        self.test_data = imputerData.ImputerData(path,
                                    files=self.test_files,
                                    batchSize=self.batchSize,
                                    isTrain=False,
                                    normalize=self.normalize,
                                    padding=self.padding,
                                    mean=self.train_data.mean,
                                    std=self.train_data.std,
                                    maxLength=self.train_data.maxLength)

    
        


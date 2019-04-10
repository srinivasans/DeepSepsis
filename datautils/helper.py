import os
import numpy as np
from shutil import copyfile
from . import dataset

def create_test_folder(data='data/challenge_data'):
    datasets = dataset.Dataset(data, train_ratio=0.8)
    test_folder = data + "_test"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for file in datasets.test_data.files:
        copyfile(os.path.join(data,file), os.path.join(test_folder, file))


def save_output(predictions, filenames, result_path, experiment, imputation_method, seed=0, threshold=0.5):
    folder = os.path.join(result_path,experiment,imputation_method,('_').join(['seed',str(seed)]))
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for i in range(0,len(predictions)):
        filename = os.path.join(folder, filenames[i])
        with open(filename,'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            for j in range(0, len(predictions[i])):
                f.write(str(predictions[i][j][0])+'|'+str(int(predictions[i][j][0]>threshold))+'\n')

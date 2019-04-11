from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from prediction.GRUDModel.models import create_grud_model, load_grud_model
from prediction.GRUDModel.callbacks import ModelCheckpointwithBestWeights

from datautils.dataset import Dataset
from datautils.data import Data


#%%
# set GPU usage for tensorflow backend
if K.backend() == 'tensorflow':
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .1
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))


#%%
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--data', default="small_challenge_data")
arg_parser.add_argument('--induce_missingness', default="false")
arg_parser.add_argument('--missing_rate', type=float, default=0.4)
arg_parser.add_argument('--generate_files', default="false")

ARGS = arg_parser.parse_args()
print('Arguments:', ARGS)

#%%
dataset = Dataset(os.path.join("data",ARGS.data),
                    batchSize=100,
                    train_ratio=0.8,
                    normalize=(True if ARGS.induce_missingness == "true" else False),
                    padding=False,
                    imputeForward=False,
                    calculateDelay=True,
                    seed=42)


#%%
import tensorflow as tf

def roc_auc_score_mod(y_true, y_pred):
    try:
        auc_val = roc_auc_score(np.reshape(y_true, (-1,1)), np.reshape(y_pred, (-1,1)))
    except ValueError:
        auc_val = 0.0
    return np.float32(auc_val)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score_mod, (y_true, y_pred), tf.float32)


#%%
model_dirs = os.listdir("saved_models")
print(f"Available models = {model_dirs}")
path = os.path.join("saved_models", model_dirs[0], "model.h5")
print(f"Using model {path}")


#%%
from prediction.GRUDModel.nn_utils import _get_scope_dict

scope_dict = _get_scope_dict()
scope_dict["auroc"] = auroc

model = keras.models.load_model(path, custom_objects=scope_dict)
#model.summary()


#%%
input_decay_kernel = model.layers[6].get_weights()[3]
input_decay_bias = model.layers[6].get_weights()[4]


#%%
# dataset.train_data.timeDelay[0][20]
# gamma_t = (input_decay_kernel * dataset.train_data.timeDelay[0][20]) + input_decay_bias
# gamma_t = np.exp(-np.maximum(0, gamma_t))
# gamma_t
# #columns = list(dataset.train_data.features.keys())[:-2]
# #[(columns[idx],gamma_t[idx]) for idx in gamma_t.argsort()]


#%%
def impute(dt: Data, mean):
    x_imputed = []

    mse = 0.0
    mse_count = 0.0

    mse_mean = 0.0

    print('-'*20)
    print("* Imputing data...")
    for i in tqdm(range(dt.x.shape[0])):
        X = dt.x[i]
        M = dt.m[i]
        if ARGS.induce_missingness == "true":
            M_induced = np.random.choice([0, 1], size=X.shape[0]*X.shape[1], p=[ARGS.missing_rate, 1.0-ARGS.missing_rate])
            M_induced = M_induced.reshape(X.shape)
            M_induced = M * M_induced
            X_new = X * M_induced
            M_current = M_induced
        else:
            X_new = X
            M_induced = M * 0
            M_current = M
        D = dt.timeDelay[i]

        X_imputed = np.zeros_like(X)
        T = X.shape[0]

        x_t_last = mean

        for t in range(T):
            x_t = X_new[t]
            d = D[t]
            m = M_current[t]
            m_inv = 1 - m

            gamma_t = (input_decay_kernel * d) + input_decay_bias
            gamma_t = np.exp(-np.maximum(0, gamma_t))

            X_imputed[t] = (m * x_t) + (m_inv * ((gamma_t * x_t_last) + ((1 - gamma_t) * mean)))

            x_t_last = (m * x_t) + (m_inv * x_t_last)

        x_imputed.append(X_imputed)

        mse += (np.square(X - X_imputed) * M).sum()
        mse_count += (M.sum() - M_induced.sum())

        if ARGS.induce_missingness == "true":
            mse_mean += (np.square(X - X_new) * M).sum()

    x_imputed = np.array(x_imputed)
    mse = mse / mse_count
    print("* Imputation complete.")
    print(f"GRUD Imputation MSE = {mse}")
    if ARGS.induce_missingness == "true":
        mse_mean = mse_mean / mse_count
        print(f"Mean Imputation MSE = {mse_mean}")
    print('-'*20)
    
    return x_imputed


#%%
def save_output(imputed,labels,filenames):
    folder = os.path.join("data",ARGS.data+"_GRUD_imputed")
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("* Generating files...")
    for i in tqdm(range(0,len(imputed))):
        filename = os.path.join(folder, filenames[i])
        with open(filename,'w') as f:
            f.write('HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel\n')
            # For each time step
            for j in range(0, len(imputed[i])):
                # For each feature
                for k in range(0,len(imputed[i][j])):
                    f.write(str(imputed[i][j][k])+'|')
                f.write('0|0|0|'+str(j)+'|'+str(labels[i][j][0])+'\n')


#%%
if ARGS.induce_missingness == "true":
    mean = np.zeros_like(dataset.train_data.mean)
else:
    mean = dataset.train_data.mean
print("Train:")
train_imputed = impute(dataset.train_data, mean)
print("Val:")
val_imputed = impute(dataset.val_data, mean)
print("Test:")
test_imputed = impute(dataset.test_data, mean)


#%%
if ARGS.generate_files == "true":
    print("Train:")
    save_output(train_imputed, dataset.train_data.y, dataset.train_data.files)
    print("Val:")
    save_output(val_imputed, dataset.val_data.y, dataset.val_data.files)
    print("Test:")
    save_output(test_imputed, dataset.test_data.y, dataset.test_data.files)
    
print('Finished!', '='*20)



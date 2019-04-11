from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import os

from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from prediction.GRUDModel.models import create_grud_model, load_grud_model
from prediction.GRUDModel.callbacks import ModelCheckpointwithBestWeights
from datautils.dataset import Dataset
from datautils.helper import save_output

#%%
# set GPU usage for tensorflow backend
if K.backend() == 'tensorflow':
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .1
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

#%%
# parse arguments
## general
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--working_path', default=os.path.join("prediction","GRUDModel"))

## data
arg_parser.add_argument('dataset_name', default='physionet',
                        help='The data files should be saved in [working_path]/data/[dataset_name] directory.')
arg_parser.add_argument('label_name', default='sepsis')
# arg_parser.add_argument('--max_timesteps', type=int, default=200, 
#                         help='Time series of at most # time steps are used. Default: 200.')
# arg_parser.add_argument('--max_timestamp', type=int, default=48*60*60,
#                         help='Time series of at most # seconds are used. Default: 48 (hours).')

## model
arg_parser.add_argument('--recurrent_dim', type=lambda x: x and [int(xx) for xx in x.split(',')] or [], default='64')
arg_parser.add_argument('--hidden_dim', type=lambda x: x and [int(xx) for xx in x.split(',')] or [], default='64')
arg_parser.add_argument('--model', default='GRUD', choices=['GRUD', 'GRUforward', 'GRU0', 'GRUsimple'])
arg_parser.add_argument('--use_bidirectional_rnn', default=False)
                           
## training
arg_parser.add_argument('--pretrained_model_file', default=None,
                        help='If pre-trained model is provided, training will be skipped.') # e.g., [model_name]_[i_fold].h5
arg_parser.add_argument('--epochs', type=int, default=100)
arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
arg_parser.add_argument('--batch_size', type=int, default=100)
arg_parser.add_argument('--impute_forward', type=int, default=0)
arg_parser.add_argument('--seed', type=int, default=42)
arg_parser.add_argument('--data', default="sepsis_data")
arg_parser.add_argument('--imputation_mode', default="decay")

## set the actual arguments if running in notebook
if not (__name__ == '__main__' and '__file__' in globals()):
    ARGS = arg_parser.parse_args([
        'physionet',
        'sepsis',
        '--model', 'GRUD',
        '--hidden_dim', '',
        '--epochs', '100'
    ])
else:
    ARGS = arg_parser.parse_args()

print('Arguments:', ARGS)

#%%
dataset = Dataset("data/"+ARGS.data,
                    batchSize=100,
                    train_ratio=0.8,
                    normalize=True,
                    padding=True,
                    imputeForward=(True if ARGS.impute_forward else False),
                    seed=ARGS.seed)

dataset.test_data.batchSize = 1

#%%
def roc_auc_score_mod(y_true, y_pred):
    try:
        auc_val = roc_auc_score(np.reshape(y_true, (-1,1)), np.reshape(y_pred, (-1,1)))
    except ValueError:
        auc_val = 0.0
    return np.float32(auc_val)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score_mod, (y_true, y_pred), tf.float32)

def remove_padding(y_true, y_pred, data):
    targets = []
    predictions = []
    predictions_ind = []
    files = []

    for i in range(y_pred.shape[0]):
        targets.extend(list(y_true[i, 0:data.x_lengths[i]]))
        predictions.extend(list(y_pred[i, 0:data.x_lengths[i]]))
        predictions_ind.append(list(y_pred[i, 0:data.x_lengths[i]]))
        files.append(data.files[i])
    
    targets = np.reshape(np.array(targets), (-1,1))
    predictions = np.reshape(np.array(predictions), (-1,1))

    return targets, predictions, predictions_ind, files

def get_auc(y_true, y_pred, data):
    targets = []
    predictions = []
    files = []

    for i in range(len(data.x_lengths)):
        targets.extend(list(y_true[i, 0:data.x_lengths[i]]))
        predictions.extend(list(y_pred[i, 0:data.x_lengths[i]]))
        files.append(data.files[i])
    
    targets = np.reshape(np.array(targets), (-1,1))
    predictions = np.reshape(np.array(predictions), (-1,1))
    
    return roc_auc_score(targets, predictions)

# k-fold cross-validation
pred_y_list_all = []
auc_score_list_all = []

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
print('Timestamp: {}'.format(timestamp))

#for i_fold in range(1):
i_fold = 0
print('{}-th fold...'.format(i_fold))

# Load or train the model.
if ARGS.pretrained_model_file is not None:
    model = load_grud_model(os.path.join(ARGS.working_path, 
                                            ARGS.pretrained_model_file.format(i_fold=i_fold)))
else:
    model = create_grud_model(input_dim=dataset.train_data.input_dim,
                                output_dim=dataset.train_data.output_dim,
                                output_activation="sigmoid",
                                recurrent_dim=ARGS.recurrent_dim,
                                hidden_dim=ARGS.hidden_dim,
                                predefined_model=ARGS.model,
                                use_bidirectional_rnn=ARGS.use_bidirectional_rnn,
                                input_decay=('exp_relu' if ARGS.imputation_mode == "decay" else None),
                                x_imputation=('raw' if ARGS.imputation_mode == "mean" else 'forward')
                                )
    if i_fold == 0:
        model.summary()
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', auroc], sample_weight_mode="temporal")
    model.fit_generator(
        generator=dataset.train_data.getBatchGenerator(shuffle=True),
        steps_per_epoch=dataset.train_data.getSteps(),
        epochs=ARGS.epochs,
        verbose=1,
        validation_data=dataset.val_data.getBatchGenerator(shuffle=True),
        validation_steps=dataset.val_data.getSteps(),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=ARGS.early_stopping_patience, restore_best_weights=True),
            ModelCheckpointwithBestWeights(
                file_dir=os.path.join(ARGS.working_path, 'model', ARGS.imputation_mode + '_seed-' + str(ARGS.seed) + '_' + timestamp + '_' + str(i_fold))
            ),
            TensorBoard(
                log_dir=os.path.join(ARGS.working_path, 'tb_logs', timestamp + '_' + str(i_fold))
            )
        ]
        )
    model.save(os.path.join(ARGS.working_path, 'model', 
                            ARGS.imputation_mode + '_seed-' + str(ARGS.seed) + '_' + timestamp + '_' + str(i_fold), 'model.h5'))

# Evaluate the model
true_y_list = [
    dataset.train_data.getTargets(), dataset.val_data.getTargets(), dataset.test_data.getTargets()
]
pred_y_list = [
    model.predict_generator(dataset.train_data.getInputGenerator(),
                            steps=dataset.train_data.getSteps()),
    model.predict_generator(dataset.val_data.getInputGenerator(),
                            steps=dataset.val_data.getSteps()),
    model.predict_generator(dataset.test_data.getInputGenerator(),
                            steps=dataset.test_data.getSteps()),
]
dataset_list = [
    dataset.train_data,
    dataset.val_data,
    dataset.test_data
]
processed_list = [remove_padding(ty, py, d) for ty, py, d in zip(true_y_list, pred_y_list, dataset_list)]
#auc_score_list = [get_auc(ty, py, d) for ty, py, d in zip(true_y_list, pred_y_list, dataset_list)] # [3, n_task]
auc_score_list = [roc_auc_score(t, p) for t,p,_,_ in processed_list] # [3, n_task]
print('AUC scores (train,val,test): {}'.format(auc_score_list))
pred_y_list_all.append(pred_y_list)
auc_score_list_all.append(auc_score_list)

auc_score_list_all = np.stack(auc_score_list_all, axis=0)
# print('Mean AUC score: {}; Std AUC score: {}'.format(
#     np.mean(auc_score_list_all, axis=0),
#     np.std(auc_score_list_all, axis=0)))

result_path = os.path.join(ARGS.working_path, 'results', timestamp)
if not os.path.exists(result_path):
    os.makedirs(result_path)
np.savez_compressed(os.path.join(result_path, 'predictions.npz'),
                    pred_y_list_all=pred_y_list_all)
np.savez_compressed(os.path.join(result_path, 'auroc_score.npz'),
                    auc_score_list_all=auc_score_list_all)

for (ty, py, _, _) in processed_list:
    print('-'*20)
    c_m = confusion_matrix(ty, np.array(py > 0.5).astype(int))
    print(c_m)
    PPV = c_m[1,1] / (c_m[1,1] + c_m[0,1])
    print(f"PPV/Precision = {PPV}")
    TPR = c_m[1,1] / c_m[1].sum()
    print(f"TPR/Sensitivity/Recall = {TPR}")
    TNR = c_m[0,0] / c_m[0].sum()
    print(f"TNR/Specificity = {TNR}")

print("Saving output...")

test_ty, test_py, test_py_ind, test_files = processed_list[2]
save_output(test_py_ind, test_files, "results", "GRU-D", ARGS.imputation_mode, seed=ARGS.seed, threshold=0.5)

print('Finished!', '='*20)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from lifelines import CoxPHFitter
from datautils.dataset import Dataset
from datautils.data import Data
from datautils.helper import save_output
from tqdm import tqdm
import argparse


#%%
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--imputation_mode', default="mean")
arg_parser.add_argument('--seed', type=int, default=42)
arg_parser.add_argument('--positive_weight', type=int, default=56)

ARGS = arg_parser.parse_args()


#%%
print(f"Running Cox with imputation_mode = {ARGS.imputation_mode}, seed = {ARGS.seed}")

print('Arguments:', ARGS)

#%%
dataset = Dataset("data/challenge_data",
                    batchSize=100,
                    train_ratio=0.8,
                    normalize=True,
                    padding=False,
                    imputeForward=(False if ARGS.imputation_mode == "mean" else True),
                    calculateDelay=False,
                    seed=ARGS.seed)


#%%
columns = list(dataset.train_data.features.keys())[:-2]

# dataset.train_data.x.shape
# dataset.val_data.x.shape
# dataset.test_data.x.shape


#%%
# create windowing system here
T = 6
#idx = 10
def process_data(d: Data, T: int) -> (pd.DataFrame, np.array):
    npa = d.x
    target_npa = d.y
    
    processed = []
    labels = []

    print("* Processing data...")
    for idx in tqdm(range(npa.shape[0])):
        if target_npa[idx].sum() == 0:
            processed.extend([[row,7,1] for row in npa[idx]])
        else:
            sepsis_count = 0
            for i in range(npa[idx].shape[0]):
                t = (T + 1) - sepsis_count
                t = t if t >= 1 else 1
                s = 1 if t > T else 0
                processed.append([npa[idx][i],t,s])
                sepsis_count += 1 if target_npa[idx][i][0] == 1 else 0
                
        labels.extend(target_npa[idx].flatten().tolist())
                
    return (pd.DataFrame(processed, columns=["x","t","s"]), np.array(labels))
# Naive windowing:
#             for i in range(df[idx].shape[0]):
#                 window = df[idx][i:i+T]
#                 matches = np.where(window[:,-1]==1)[0]
#                 if matches.size > 0:
#                     t = matches[0] + 1
#                     s = 0
#                 else:
#                     t = T + 1
#                     s = 1
#                 processed.append([df[idx][i][:-1],t,s])


#%%
X_train, y_train = process_data(dataset.train_data, T)
X_val, y_val = process_data(dataset.val_data, T)
X_test, y_test = process_data(dataset.test_data, T)


#%%
# X_train.head()


#%%
inverse_s = 1-X_train.s
X_train_cph = pd.DataFrame(X_train.x.values.tolist(), columns=columns)
X_train_cph["s"] = inverse_s
X_train_cph["w"] = (inverse_s * ARGS.positive_weight) + X_train.s
X_train_cph["t"] = X_train.t


#%%
cph = CoxPHFitter(penalizer=0.2)
cph.fit(X_train_cph, duration_col='t', event_col='s', weights_col='w', step_size=0.070, show_progress=True, robust=False)


#%%
#cph.check_assumptions(X_train_cph,show_plots=False,plot_n_bootstraps=0)
#cph.print_summary()


#%%
def get_metrics(ty, py, threshold=0.5):
    print('-'*20)
    auc = roc_auc_score(ty, py)
    print(f"AUC = {auc}")
    lst = [1 if i >=0.5 else 0 for i in py]
    acc = ((lst == ty).sum() / ty.shape[0]) * 100
    print(f"Accuracy = {acc}")
    c_m = confusion_matrix(ty, np.array(py > threshold).astype(int))
    print(c_m)
    PPV = c_m[1,1] / (c_m[1,1] + c_m[0,1])
    print(f"PPV/Precision = {PPV}")
    TPR = c_m[1,1] / c_m[1].sum()
    print(f"TPR/Sensitivity/Recall = {TPR}")
    TNR = c_m[0,0] / c_m[0].sum()
    print(f"TNR/Specificity = {TNR}")
    print('-'*20)


#%%
def get_preds(df: pd.DataFrame, columns):
    cph_df = pd.DataFrame(df.x.values.tolist(), columns=columns)
    
    preds = 1-cph.predict_survival_function(cph_df,times=[6])
    
    return preds


#%%
print("Train:")
train_preds = get_preds(X_train, columns)
get_metrics(y_train, train_preds, threshold=0.5)
print("Val:")
val_preds = get_preds(X_val, columns)
get_metrics(y_val, val_preds, threshold=0.5)
print("Test:")
test_preds = get_preds(X_test, columns)
get_metrics(y_test, test_preds, threshold=0.5)


#%%
test_preds = test_preds.values


#%%
grouped_preds = []

cur = 0
for x_length in dataset.test_data.x_lengths:
    grouped_preds.append(list(test_preds[cur:cur+x_length].reshape((-1,1))))
    cur += x_length


#%%
save_output(grouped_preds, list(dataset.test_data.files), "results", "COX", ARGS.imputation_mode, seed=ARGS.seed, threshold=0.5)

print('Finished!', '='*20)

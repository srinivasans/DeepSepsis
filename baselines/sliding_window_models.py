import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datautils import dataset
import numpy as np 
import pandas as pd
import sklearn 
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve
import multiprocessing as mp 

def createSWData(X, y, window_size):
    X_train = []
    y_train = []

    for i in range(X.shape[0]):
        # Get patient data
        X_patient = X[i]
        y_patient = y[i].ravel()

        # Get sliding-window data and append to X_train
        window_start = 0
        while (window_start + window_size) <= X_patient.shape[0]:
            window_data = X_patient[window_start:window_start + window_size, :]
            assert window_data.shape[0] == window_size

            x_i = np.reshape(window_data, (window_size*window_data.shape[1],))
            X_train.append(x_i)
            y_train.append(y_patient[window_start + window_size - 1])

            window_start += 1
    
    return np.array(X_train), np.array(y_train)

def utility_train_predict(model, model_type, data, window_size, impute_method, seed):
    # Train model 
    if model_type == 'RLR':
        X_train, y_train = createSWData(np.abs(data.train_data.x), data.train_data.y, window_size)
    else:
        X_train, y_train = createSWData(data.train_data.x, data.train_data.y, window_size)

    model = model.fit(X_train, y_train)

    # Create test files for utility scores
    X_test, y_test, files = data.test_data.x, data.test_data.y, data.test_data.files
    for i in range(X_test.shape[0]):
        X_sw = []
        X_patient = X_test[i]
        y_sw = y_test[i]
        
        # Append rows of values to top of df 
        mean_vals = np.mean(X_patient, axis=0)
        X_patient = np.concatenate(([mean_vals]*max(1,window_size-1), X_patient))
    
        # Get data for every window
        window_start = 0
        while (window_start + window_size) <= X_patient.shape[0]:
            window_data = X_patient[window_start:window_start + window_size,:]
            assert window_data.shape[0] == window_size

            x_i = np.reshape(window_data, (window_size*window_data.shape[1],))
            X_sw.append(x_i)

            window_start += 1
        
        # Predict for every time step
        y_pred = model.predict(X_sw)
        y_pred_prob = model.predict_proba(X_sw)
        
        assert y_pred.shape[0] == y_sw.shape[0]
        
        # Save to file
        if not os.path.isdir('results/%s/%s/seed_%d'%(model_type, impute_method, seed)):
            os.makedirs('results/%s/%s/seed_%d'%(model_type, impute_method, seed))
            
        pred = np.transpose(np.vstack((y_pred_prob[:,1], y_pred)))
        pred_df = pd.DataFrame(pred, columns=["PredictedProbability", "PredictedLabel"])
        pred_df.to_csv('results/%s/%s/seed_%d/%s'%(model_type, impute_method, seed, files[i]), sep='|', index=False)

def train_predict(model, model_label, data, impute_method):

    # ws, imp. method, train score, train recall, train prec, test score, 
    # test auc, test recall, test prec, test conf matrix
    results = []

    for ws in range(1,7):
        res = [ws, impute_method]
        print("Working on ws: %d, imp: %s"%(ws,impute_method))

        if model_label == 'RLR':
            X_train, y_train = createSWData(np.abs(data.train_data.x), data.train_data.y, ws)
        else:
            X_train, y_train = createSWData(data.train_data.x, data.train_data.y, ws)

        trained_model = model.fit(X_train, y_train)

        res.append(trained_model.score(X_train, y_train))
        y_pred = trained_model.predict(X_train)
        res.extend([recall_score(y_train, y_pred), precision_score(y_train, y_pred)])

        X_test, y_test = createSWData(data.val_data.x, data.val_data.y, ws)
        res.append(trained_model.score(X_test, y_test))

        y_pred_prob = trained_model.predict_proba(X_test)
        res.append(roc_auc_score(y_test, y_pred_prob[:,1]))

        y_pred = trained_model.predict(X_test)
        res.extend([recall_score(y_test, y_pred), precision_score(y_test, y_pred)])
        res.extend(confusion_matrix(y_test, y_pred).ravel())
            
        results.append(res)
            
    return np.array(results)

def save_results(res, path):
    with open(path, 'w') as f:
        f.write("WS, Imp, train acc, train recall, train prec, test acc, test auc, test recall, test prec, conf matrix\n")
        for i in range(res.shape[0]):
            f.write( " ".join([str(v) for v in res[i,:]]) + '\n')

# Regularized Logistic Regression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
def run_rlr(utility_predict=False, rand_seed=None):
    rlr_model = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500, class_weight='balanced')
    if not utility_predict:
        rlr_mean_res = train_predict(rlr_model, 'RLR', datasets_mean, 'mean')
        rlr_forw_res = train_predict(rlr_model, 'RLR', datasets_forw, 'forw')
        save_results(rlr_mean_res, 'baselines/RLR_mean')
        save_results(rlr_forw_res, 'baselines/RLR_forw')
    else:
        utility_train_predict(rlr_model, 'RLR', datasets_forw, 5, 'forw', rand_seed)
        utility_train_predict(rlr_model, 'RLR', datasets_mean, 4, 'mean', rand_seed)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier
def run_rf(utility_predict=False, rand_seed=None):
    rf_model = RandomForestClassifier(n_estimators=25, max_depth=5, class_weight='balanced', n_jobs=-1)
    if not utility_predict:
        rf_mean_res = train_predict(rf_model, 'RF', datasets_mean, 'mean')
        rf_forw_res = train_predict(rf_model, 'RF', datasets_forw, 'forw')
        save_results(rf_mean_res, 'baselines/RF_mean')
        save_results(rf_forw_res, 'baselines/RF_forw')
    else:
        print("Running utility predictions for RF with ws=%d, imp=%s, seed=%d"%(5, 'mean', rs))
        utility_train_predict(rf_model, 'RF', datasets_forw, 6, 'forw', rand_seed)
        utility_train_predict(rf_model, 'RF', datasets_mean, 5, 'mean', rand_seed)

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier 
def train_predict_xgb(model, data, impute_method):
    results = []

    for ws in range(1,7):
        res = [ws, impute_method]
        print("Working on ws: %d, imp: %s"%(ws,impute_method))

        # Get Data
        X_train, y_train = createSWData(data.train_data.x, data.train_data.y, ws)
        X_val, y_val = createSWData(data.val_data.x, data.val_data.y, ws)

        # Train model
        # evallist = evallist = [(dtrain, 'train'), (dval, 'eval')]
        bst = model.fit(X_train, y_train)

        # Training metrics
        y_pred = bst.predict(X_train)
        train_acc = 1 - np.sum(np.abs(y_train - y_pred))/y_train.shape[0]
        res.extend([train_acc, recall_score(y_train, y_pred), precision_score(y_train, y_pred)])

        # Test metrics
        y_pred_prob = bst.predict_proba(X_val)
        y_pred = bst.predict(X_val)
        res.append(roc_auc_score(y_val, y_pred_prob[:,1]))
        res.extend([recall_score(y_val, y_pred), precision_score(y_val, y_pred)])
        res.extend(confusion_matrix(y_val, y_pred).ravel())
            
        results.append(res)
            
    return np.array(results)

def run_xgb(utility_predict=False, rand_seed=None):
    xgb_model = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=1, reg_lambda=1000, scale_pos_weight=55.6, n_jobs=-1)
    if not utility_predict:
        xgb_mean_res = train_predict_xgb(xgb_model, datasets_mean, 'mean')
        xgb_forw_res = train_predict_xgb(xgb_model, datasets_forw, 'forw')
        save_results(xgb_mean_res, 'baselines/XG_mean')
        save_results(xgb_forw_res, 'baselines/XG_forw')
    else:
        utility_train_predict(xgb_model, 'XG', dataset_mean, 6, 'mean', rand_seed)
        utility_train_predict(xgb_model, 'XG', dataset_forw, 5, 'forw', rand_seed)


# AdaBoost
# THIS NEEDS TO BE Tuned ... 
from sklearn.ensemble import GradientBoostingClassifier
def run_adb(utility_predict=False):
    ab_model = GradientBoostingClassifier(n_estimators=25, loss='exponential')
    ab_mean_res = train_predict(ab_model, 'AB', datasets_mean, 'mean')
    ab_forw_res = train_predict(ab_model, 'AB', datasets_forw, 'forw')
    save_results(ab_mean_res, 'baselines/AB_mean')
    save_results(ab_forw_res, 'baselines/AB_forw')

# SVM
from sklearn.svm import SVC
def run_svm(utility_predict=False):
    svm_model = SVC(C=0.001, gamma='auto', class_weight='balanced', probability=True)
    svm_mean_res = train_predict(svm_model, "SVM", datasets_mean, 'mean')
    svm_forw_res = train_predict(svm_model, "SVM", datasets_forw, 'forw')
    save_results(svm_mean_res, 'baselines/SVM_mean')
    save_results(svm_forw_res, 'baselines/SVM_forw')

# Neural Network
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

def run_nn(data, ws, imp):
    res = [ws, imp]

    nn_model = Sequential()
    nn_model.add(Dense(432, activation='relu', input_dim=ws*36))
    nn_model.add(Dropout(rate=0.5))
    nn_model.add(Dense(216, activation='relu'))
    nn_model.add(Dropout(rate=0.5))
    nn_model.add(Dense(108, activation='relu'))
    nn_model.add(Dropout(rate=0.5))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(optimizer='Adam', loss='binary_crossentropy')

    X_train, y_train = createSWData(data.train_data.x, data.train_data.y, ws)
    X_val, y_val = createSWData(data.val_data.x, data.val_data.y, ws)

    cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    early_stop = EarlyStopping(patience=5)
    nn_model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_val, y_val), 
                    class_weight=cw, callbacks=[early_stop])
    
    X_test, y_test = createSWData(data.test_data.x, data.test_data.y, ws)
    y_pred = nn_model.predict(X_test)
    res.append(roc_auc_score(y_test, y_pred))
    res.extend([recall_score(y_test, (y_pred>0.5)), precision_score(y_test, (y_pred>0.5))])
    res.extend(confusion_matrix(y_test, (y_pred>0.5)).ravel())

    res = np.reshape(np.array(res), (1,len(res)))
    save_results(res, 'baselines/NN_%s'%imp)

# Get Data and Run Models 
random_seeds = [1, 21, 23, 30]
impute_methods = ['mean', 'forward', 'DAE', 'kNN', "GRU-D"]
# datasets_mean = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, padding=False, calculateDelay=False)
# datasets_forw = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, imputeForward=True, calculateDelay=False, padding=False)

# print("Running RLR..")
# run_rlr()
# print("Running RF..")
# run_rf()
# print("Running AB..")
# run_adb(utility_predict=True)
# print("Running XGB..")
# run_xgb()
# print("Running SVM..")
# run_svm(utility_predict=False)
# print("Running NN..")
# run_nn(datasets_mean, 6, 'mean')
# run_nn(datasets_forw, 6, 'forw')

# Run utility predictions
for rs in random_seeds:
    datasets_mean = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, padding=False, calculateDelay=False, seed=rs)
    datasets_forw = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, imputeForward=True, calculateDelay=False, padding=False, seed=rs)

    print("Running RLR Utility..")
    run_rlr(utility_predict=True, rs)
    print("Running RF Utility..")
    run_rf(utility_predict=True, rs)
    print("Running XGB Utility..")
    run_xgb(utility_predict=True, rs)


























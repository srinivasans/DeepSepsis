import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datautils import dataset
import numpy as np 
import sklearn 
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve

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

def utility_predict(model, model_type, test_data, window_size, impute_method):
    X_test, y_test, files = test_data.x, test_data.y, test_data.test_files
    for i in range(X_test.shape[0]):
        X_sw = []
        y_sw = []
        X_patient = X_test[i]
        y_patient = y_test[i]
        
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
            y_sw.append(y_patient[window_start+window_size-1])

            window_start += 1
        
        # Predict for every time step
        y_pred = model.predict(X_sw)
        y_pred_prob = model.predict_proba(X_sw)
        
        assert y_pred.shape[0] == y_sw.shape[0]
        
        # Save to file
        if not os.path.isdir('results/%s/%s_%d'%(model_type, impute_method, window_size)):
            os.mkdir('results/%s/%s_%d'%(model_type, impute_method, window_size))
            
        pred = np.transpose(np.vstack((y_pred_prob[:,1], y_pred)))
        pred_df = pd.DataFrame(pred, columns=["PredictedProbability", "PredictedLabel"])
        pred_df.to_csv('results/%s/%s_%d/%s.psv'%(model_type, impute_method, window_size, files[i]), sep='|', index=False)

def train_predict(model, model_label, data, impute_method):

    # ws, imp. method, train score, train recall, train prec, test score, 
    # test auc, test recall, test prec, test conf matrix
    results = []

    for ws in range(1,7):
        res = [ws, impute_method]
        print("Working on ws: %d, imp: %s"%(ws,impute_method))

        X_train, y_train = createSWData(np.abs(data.train_data.x), data.train_data.y, ws)
        model = model.fit(X_train, y_train)
        # utility_predict(model, model_label, data.test_data, ws, impute_method)

        res.append(model.score(X_train, y_train))
        y_pred = model.predict(X_train)
        res.extend([recall_score(y_train, y_pred), precision_score(y_train, y_pred)])

        X_test, y_test = createSWData(data.val_data.x, data.val_data.y, ws)
        res.append(model.score(X_test, y_test))

        y_pred_prob = model.predict_proba(X_test)
        res.append(roc_auc_score(y_test, y_pred_prob[:,1]))

        y_pred = model.predict(X_test)
        res.extend([recall_score(y_test, y_pred), precision_score(y_test, y_pred)])
        res.extend(confusion_matrix(y_test, y_pred).ravel())
            
        results.append(res)
            
    return np.array(results)

def save_results(res, path):
    with open(path, 'a') as f:
        for i in range(res.shape[0]):
            f.write( " ".join([str(v) for v in res[i,:]]) + '\n')

random_seed = [1, 21, 23, 30]
impute_methods = ['mean', 'forward', 'DAE', 'kNN', "GRU-D"]
datasets_mean = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, padding=False)
datasets_forw = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336, imputeForward=True, padding=False)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
lr_model = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
rlr_mean_res = train_predict(lr_model, 'RLR', datasets_mean, 'mean')
rlr_forw_res = train_predict(lr_model, 'RLR', datasets_forw, 'forw')
save_results(rlr_mean_res, 'baselines/RLR_mean')
save_results(rlr_forw_res, 'baselines/RLR_forw')

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=5, class_weight='balanced')
rf_mean_res = train_predict(rf_model, 'RF', datasets_mean, 'mean')
rf_forw_res = train_predict(rf_model, 'RF', datasets_forw, 'forw')
save_results(rf_mean_res, 'baselines/RF_mean')
save_results(rf_forw_res, 'baselines/RF_forw')

# XGBoost
import xgboost as xgb 
dtrain = xgb.DMatrix(datasets_mean.train_data.x, label=datasets_mean.train_data.y)


# AdaBoost TODO
# from sklearn.ensemble import GradientBoostingClassifier

# ab_model = GradientBoostingClassifier(loss='exponential')

# import keras
# from keras.models import Sequential 
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
# from sklearn.utils import class_weight

# nn_results = []

# for ws in range(3,7):
#     nn_model = Sequential()
#     nn_model.add(Dense(128, activation='relu', input_dim=ws*34))
#     nn_model.add(Dropout(rate=0.5))
#     nn_model.add(Dense(64, activation='relu'))
#     nn_model.add(Dropout(rate=0.5))
#     nn_model.add(Dense(32, activation='relu'))
#     nn_model.add(Dense(1, activation='sigmoid'))
#     nn_model.compile(optimizer='Adam', loss='binary_crossentropy')
    
#     for imp in ['mean', 'forw']:
#         res = [ws, imp]
#         X_train, y_train, _ = create_train_data(ws, imp)
#         cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#         nn_model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.3, class_weight=cw)
        
#         X_test, y_test = predict(nn_model, "NN", ws, imp)
#         y_pred = nn_model.predict(X_test)
#         res.append(roc_auc_score(y_test, y_pred))
#         res.extend([recall_score(y_test, (y_pred>0.5)), precision_score(y_test, (y_pred>0.5))])
#         res.extend(confusion_matrix(y_test, (y_pred>0.5)).ravel())
        
#         nn_results.append(res)

# nn_results = np.array(nn_results)
# nn_results.tofile('../results/nn_results', sep='\n')




























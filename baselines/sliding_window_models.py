import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datautils import dataset
import numpy as np 
import sklearn 
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve

def create_train_val_data(window_size, impute_method):
    X_train = []
    y_train = []
    features = None

    i = 0
    for file in os.listdir('../data/train_%s_imputed/'%impute_method):

        # Read file
        patient_df = []
        with open('../data/train_%s_imputed/%s' % (impute_method, file)) as f:

            if not features:
                features = f.readline().rstrip('\n').split('|')
            else:
                # This skips the headers
                f.readline()

            for idx, line in enumerate(f):
                line = line.rstrip('\n')
                patient_df.append(line.split('|'))

        patient_df = np.array(patient_df)
        
        # Get sliding-window data
        window_start = 0
        while (window_start + window_size) <= patient_df.shape[0]:
            window_data = patient_df[window_start:window_start + window_size, 0:34].astype(np.float)
            assert window_data.shape[0] == window_size

            x_i = np.reshape(window_data, (window_size*window_data.shape[1],))
            X_train.append(x_i)

            label = int(patient_df[window_start + window_size - 1, 40])
            y_train.append(label)

            window_start += 1
    
    return np.array(X_train), np.array(y_train), features

# Read test files, create windows, and predict. 

def create_test_data(window_size, impute_method):
    X_test = []
    y_test = []
    features = None
    
    for file in os.listdir('../data/test_%s_imputed/'%(impute_method)):
        X = []
        y = []
        
        # Read file 
        patient_df = []
        with open('../data/test_%s_imputed/%s' % (impute_method, file)) as f:

            if not features:
                features = f.readline().rstrip('\n').split('|')
            else:
                # This skips the headers
                f.readline()

            for idx, line in enumerate(f):
                line = line.rstrip('\n') 
                patient_df.append(line.split('|'))
        
        # Create patient df
        patient_df = np.array(patient_df)
        X_patient = np.concatenate((patient_df[:,0:34], patient_df[:,40][:,None]), axis=1).astype(np.float)
        
        assert X_patient.shape[0] == patient_df.shape[0]
        
        # Append 5 rows of mean values to top of df 
        mean_vals = np.mean(X_patient, axis=0)
        X_patient = np.concatenate(([mean_vals]*max(1,window_size-1), X_patient))
    
        # Get data for every window
        window_start = 0
        while (window_start + window_size) <= X_patient.shape[0]:
            window_data = X_patient[window_start:window_start + window_size, 0:X_patient.shape[1]-1]
            assert window_data.shape[0] == window_size

            x_i = np.reshape(window_data, (window_size*window_data.shape[1],))
            X.append(x_i)
            
            label = int(X_patient[window_start+window_size-1, X_patient.shape[1]-1])
            y.append(label)

            window_start += 1
        
        # Add X to X_test
        X_test.extend(list(X))
        y_test.extend(list(y))
        
    return np.array(X_test), np.array(y_test)

def predict(model, model_type, window_size, impute_method):

    X_test = []
    y_test = []
    features = None
    
    for file in os.listdir('../data/test_%s_imputed/'%(impute_method)):
        X = []
        y = []
        
        # Read file 
        patient_df = []
        with open('../data/test_%s_imputed/%s' % (impute_method, file)) as f:

            if not features:
                features = f.readline().rstrip('\n').split('|')
            else:
                # This skips the headers
                f.readline()

            for idx, line in enumerate(f):
                line = line.rstrip('\n') 
                patient_df.append(line.split('|'))
        
        # Create patient df
        patient_df = np.array(patient_df)
        X_patient = np.concatenate((patient_df[:,0:34], patient_df[:,40][:,None]), axis=1).astype(np.float)
        
        assert X_patient.shape[0] == patient_df.shape[0]
        
        # Append 5 rows of mean values to top of df 
        mean_vals = np.mean(X_patient, axis=0)
        X_patient = np.concatenate(([mean_vals]*max(1,window_size-1), X_patient))
    
        # Get data for every window
        window_start = 0
        while (window_start + window_size) <= X_patient.shape[0]:
            window_data = X_patient[window_start:window_start + window_size, 0:X_patient.shape[1]-1]
            assert window_data.shape[0] == window_size

            x_i = np.reshape(window_data, (window_size*window_data.shape[1],))
            X.append(x_i)
            
            label = int(X_patient[window_start+window_size-1, X_patient.shape[1]-1])
            y.append(label)

            window_start += 1
        
        # Add X to X_test
        X_test.extend(list(X))
        y_test.extend(list(y))
        
        # Predict for every time step
        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)
        
        assert y_pred.shape[0] == len(y)
        
        # Save to file
        if not os.path.isdir('../data/model_predictions/%s_pred_%s_imputed_%d'%(model_type, impute_method, window_size)):
            os.mkdir('../data/model_predictions/%s_pred_%s_imputed_%d'%(model_type, impute_method, window_size))
            print("Created new directory for model:%s, imp:%s, ws:%d"%(model_type, impute_method, window_size))
            
        pred = np.transpose(np.vstack((y_pred_prob[:,1], y_pred)))
        pred_df = pd.DataFrame(pred, columns=["PredictedProbability", "PredictedLabel"])
        pred_df.to_csv('../data/model_predictions/%s_pred_%s_imputed_%d/%s'%(model_type, impute_method, window_size, file), 
                       sep='|', index=False)
        
    return np.array(X_test), np.array(y_test)

def train_predict(model, model_label):

    # Model Label, WS, imp. method, train score, train recall, train prec, test score, test auc, test recall, test prec
    results = []

    for ws in range(1,7):
        for imp in ['mean', 'forw']:
            res = [ws, imp]
            print("Working on ws: %d, imp: %s"%(ws,imp))

            X_train, y_train, _ = create_train_data(window_size=ws, impute_method=imp)
            model = model.fit(X_train, y_train)

            res.append(model.score(X_train, y_train))
            y_pred = model.predict(X_train)
            res.extend([recall_score(y_train, y_pred), precision_score(y_train, y_pred)])

            X_test, y_test = predict(model, model_label, window_size=ws, impute_method=imp)
            res.append(model.score(X_test, y_test))

            y_pred_prob = model.predict_proba(X_test)
            res.append(roc_auc_score(y_test, y_pred_prob[:,1]))

            y_pred = model.predict(X_test)
            res.extend([recall_score(y_test, y_pred), precision_score(y_test, y_pred)])
            res.extend(confusion_matrix(y_test, y_pred).ravel())
                
            results.append(res)
            
    return np.array(results)


datasets = dataset.Dataset('../sepsis_data/all_data', train_ratio=0.8, maxLength=336)

X_train, y_train = datasets.train_data.x, datasets.train_data.y 
train_lengths = datasets.train_data.x_lengths

X_valid, y_valid = datasets.val_data.x, datasets.val_data.y
val_lengths = datasets.val_data.x_lengths

X_test, y_test = datasets.test_data.x, datasets.test_data.y
test_lengths = datasets.test_data.x_lengths

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

lr_model = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=1000, class_weight='balanced')
lr_results = train_predict(lr_model, 'RLR')
lr_results.tofile('../results/lr_results', sep='\n')

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_depth=5, class_weight='balanced')
rf_results = train_predict(rf_model, 'RF')
rf_results.tofile('../results/rf_results', sep='\n')

import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

nn_results = []

for ws in range(3,7):
    nn_model = Sequential()
    nn_model.add(Dense(128, activation='relu', input_dim=ws*34))
    nn_model.add(Dropout(rate=0.5))
    nn_model.add(Dense(64, activation='relu'))
    nn_model.add(Dropout(rate=0.5))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(optimizer='Adam', loss='binary_crossentropy')
    
    for imp in ['mean', 'forw']:
        res = [ws, imp]
        X_train, y_train, _ = create_train_data(ws, imp)
        cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        nn_model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.3, class_weight=cw)
        
        X_test, y_test = predict(nn_model, "NN", ws, imp)
        y_pred = nn_model.predict(X_test)
        res.append(roc_auc_score(y_test, y_pred))
        res.extend([recall_score(y_test, (y_pred>0.5)), precision_score(y_test, (y_pred>0.5))])
        res.extend(confusion_matrix(y_test, (y_pred>0.5)).ravel())
        
        nn_results.append(res)

nn_results = np.array(nn_results)
nn_results.tofile('../results/nn_results', sep='\n')


from sklearn.ensemble import GradientBoostingClassifier

ab_model = GradientBoostingClassifier(loss='exponential')





























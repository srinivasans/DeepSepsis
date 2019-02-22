'''
Baseline logistic regression model for NaNr
'''
import os
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

data = {}
features = []
def get_data(folder):
    files = os.listdir(folder)
    for patient_file in files:
        patient = patient_file.split('.')[0]
        data[patient] = []
        with open(os.path.join(folder,patient_file),'r') as f:
            if(len(data) <= 1):
                line = f.readline()
                feat = [str(pt) for pt in line.strip().split('|')]
                features.extend(feat)
            else:
                next(f)
            for line in f:
                point = [float(pt) for pt in line.strip().split('|')]
                data[patient].append(point)


def run_model():
    X_data = np.zeros((len(data), 34))
    Y_data = np.zeros((len(data), 1))
    idx = 0
    for key in data.keys():
        feat = np.matrix(data[key])
        feat = feat[np.where(feat[:,-1]==0)[0]]
        feat = np.nanmean(feat, axis=0)
        feat = np.nan_to_num(feat)
        X_data[idx, :] = feat[0,0:34]
        Y_data[idx, 0] = np.max(np.matrix(data[key])[:,-1])
        idx+=1
        
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=0)
    
    clf = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs',multi_class='multinomial')
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(clf, X_data, Y_data, cv=cv, scoring="roc_auc")
    print(scores)

    clf.fit(X_train, y_train)

    idxs = np.where(y_test==0)[0]
    test = X_test[idxs]
    test_y = y_test[idxs]
    y_pred_prob = clf.predict_proba(test)[:,1]
    print("Accuracy Score : ", accuracy_score(test_y, (y_pred_prob>0.5).astype(np.uint8)))
    fpr, tpr, _ = roc_curve(test_y, y_pred_prob)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.show()
    print("AUC Score : ", roc_auc_score(test_y, y_pred_prob))

        
if __name__ == '__main__':
    get_data('../sepsis_data/')
    run_model()

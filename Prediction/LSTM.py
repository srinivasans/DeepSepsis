from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.models import Sequential, load_model
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import time
import sys
sys.path.append("../DeepSepsis")
from datautils import readData


def create_model(specs):
    print("Creating model...")
    m = Sequential()
    masking_layer = Masking(mask_value=0., input_shape=(None, specs['features']))
    lstm_layer_1 = LSTM(specs['hidden_units'], input_shape=(None, specs['features']), return_sequences=True)
    # lstm_layer_2 = LSTM(64, input_shape=(None, features), return_sequences=True)
    dense_layer = TimeDistributed(Dense(1, activation='sigmoid'))

    m.add(masking_layer)
    m.add(lstm_layer_1)
    # m.add(lstm_layer_2)
    m.add(dense_layer)

    m.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print("Finshed creating model")
    return m

def train_model(x_train, y_train, specs):
    m = create_model(specs)
    count = 0
    for epoch in range(specs['epochs']):
        epoch_start = time.time()
        for i in range(x_train.shape[0]):
            if count % 100 == 0:
                print("Training example {}".format(count))
            time_steps = x_train.shape[1]
            seq = np.array(x_train[i,:]).reshape(1, time_steps, specs['features'])
            label = np.array(y_train[i]).reshape(1, time_steps, 1)
            m.train_on_batch(seq, label)
            count += 1

        epoch_end = time.time()
        print("Epoch finished in {} minutes".format((epoch_end-epoch_start)/50))

    print("Saving model...")
    m.save("lstm_model_{}.hdf5".format(specs['epochs']))
    print("Finished saving model")

def fit_model(train_data, test_data, test_xlen, specs):
    m = create_model(specs)
    m.fit(x=train_data.x, y=train_data.y, epochs=specs['epochs'], validation_data=(test_data.x, test_data.y))

    m.save("lstm_model_{}_{}_{}.hdf5".format(specs['epochs'], specs['num_layers'], specs['hidden_units']))

    pred_prob = m.predict(test_data.x)

    target = []
    predictions = []

    for i in range(0, len(test_xlen)):
        target.extend(list(test_data.y[i, 0:test_xlen[i]]))
        predictions.extend(list(pred_prob[i, 0:test_xlen[i]]))

    auc = metrics.roc_auc_score(np.array(target), np.array(predictions))
    predictions = np.array(np.array(predictions) > 0.5).astype(int)
    acc = metrics.accuracy_score(np.array(target), np.array(predictions))

    print("auc: {} acc {}".format(auc, acc))


def evaluate_model(x_test, y_test, test_xlen, model_path, specs):
    print("Evaluating model...")
    m = load_model(model_path)
    count = 0

    pred_prob = np.zeros((y_test.shape[0], y_test.shape[1]))

    for i in range(x_test.shape[0]):
        if count % 100 == 0:
            print("Predicting example {}".format(count))
        time_steps = x_test.shape[1]
        prediction = np.array(x_test[i,:]).reshape(1,time_steps,specs['features'])
        pred = m.predict_on_batch(prediction).reshape(1, time_steps)
        pred_prob[i, :] = pred
        count += 1

    target = []
    predictions = []

    for i in range(0, len(test_xlen)):
        target.extend(list(y_test[i, 0:test_xlen[i]]))
        predictions.extend(list(pred_prob[i, 0:test_xlen[i]]))

    auc = metrics.roc_auc_score(np.array(target), np.array(predictions))
    predictions = np.array(np.array(predictions) > 0.5).astype(int)
    acc = metrics.accuracy_score(np.array(target), np.array(predictions))


    print("auc: {} acc {}".format(auc, acc))


def main():
    # cur_dir = os.getcwd()
    # train_data_dir = cur_dir + "/../data/train_mean_imputed/"
    # test_data_dir = cur_dir + "/../data/test_mean_imputed/"
    #
    # y_train = []
    # x_train = []
    #
    #
    # for subdir, dirs, files in os.walk(train_data_dir):
    #     for file in files:
    #         fullFile = os.path.join(subdir, file)
    #         split_by_line = open(fullFile).read().split('\n')
    #         to_csv = []
    #         for line in split_by_line:
    #             to_csv.append(line.split('|'))
    #         data = pd.DataFrame(to_csv[1:-1], columns=to_csv[0])
    #         y_train.append(data['SepsisLabel'].tolist())
    #         data_wo_labels = data.drop(['SepsisLabel', 'pid'], axis=1)
    #         x_train.append(data_wo_labels.values)
    #
    # y_test = []
    # x_test = []
    #
    #
    # for subdir, dirs, files in os.walk(test_data_dir):
    #     for file in files:
    #         fullFile = os.path.join(subdir, file)
    #         split_by_line = open(fullFile).read().split('\n')
    #         to_csv = []
    #         for line in split_by_line:
    #             to_csv.append(line.split('|'))
    #         data = pd.DataFrame(to_csv[1:-1], columns=to_csv[0])
    #         y_test.append(data['SepsisLabel'].tolist())
    #         data_wo_labels = data.drop(['SepsisLabel', 'pid'], axis=1)
    #         x_test.append(data_wo_labels.values)

    specs = {}
    specs['epochs'] = 50
    specs['features'] = 36
    specs['num_layers'] = 1
    specs['hidden_units'] = 100


    train_data = readData.ReadData('../data/train')
    test_data = readData.ReadData('../data/test', isTrain=False, mean=train_data.mean, std=train_data.std, maxLength=train_data.maxLength)
    # train_model(train_data.x, train_data.y, specs)

    # print(test_data.y)
    # evaluate_model(test_data.x, test_data.y, test_data.x_lengths, "lstm_model_{}.hdf5".format(specs['epochs']), specs)

    fit_model(train_data, test_data, test_data.x_lengths, specs)





    # data = pd.read_csv(train_data_dir).head()


    # pids = set(data['pid'])
    # y_train = []
    # x_train = []
    #
    # for pid in pids:
    #     pid_data = data[data['pid'] == pid]
    #     y_train.append(pid_data['SepsisLabel'].tolist())
    #     pid_data_without_labels = pid_data.drop(['SepsisLabel', 'Unnamed: 0', 'pid'],
    #                   axis=1)
    #     x_train.append(pid_data_without_labels.values)



if __name__ == '__main__':
    main()

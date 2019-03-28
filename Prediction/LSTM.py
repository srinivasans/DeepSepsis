from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os


def create_model(features):
    print("Creating model...")
    m = Sequential()
    lstm_layer = LSTM(64, input_shape=(None, features), return_sequences=True)
    dense_layer = TimeDistributed(Dense(1, input_shape=(None, features), activation='sigmoid'))

    m.add(lstm_layer)
    m.add(dense_layer)

    m.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print("Finshed creating model")
    return m

def train_model(x_train, y_train, specs):
    m = create_model(specs['features'])
    count = 0
    for epoch in range(specs['epochs']):
        for seq, label in zip(x_train, y_train):
            if count % 1000 == 0:
                print("Training example {}".format(count))

            time_steps = len(seq)
            seq = np.array(seq).reshape(1, time_steps, specs['features'])
            label = np.array(label).reshape(1, time_steps)
            m.train_on_batch(seq, label)
            count += 1

    print("Saving model...")
    m.save("lstm_model_{}.hdf5".format(specs['epochs']))
    print("Finished saving model")


def main():
    cur_dir = os.getcwd()
    train_data_dir = cur_dir + "/../data/train_mean_imputed/"
    test_data_dir = cur_dir + "/../data/test_mean_imputed/"

    y_train = []
    x_train = []


    for subdir, dirs, files in os.walk(train_data_dir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            split_by_line = open(fullFile).read().split('\n')
            to_csv = []
            for line in split_by_line:
                to_csv.append(line.split('|'))
            data = pd.DataFrame(to_csv[1:-1], columns=to_csv[0])
            y_train.append(data['SepsisLabel'].tolist())
            data_wo_labels = data.drop(['SepsisLabel', 'pid', ''], axis=1)
            x_train.append(data_wo_labels.values)

    y_test = []
    x_test = []


    for subdir, dirs, files in os.walk(test_data_dir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            split_by_line = open(fullFile).read().split('\n')
            to_csv = []
            for line in split_by_line:
                to_csv.append(line.split('|'))
            data = pd.DataFrame(to_csv[1:-1], columns=to_csv[0])
            y_test.append(data['SepsisLabel'].tolist())
            data_wo_labels = data.drop(['SepsisLabel', 'pid', ''], axis=1)
            x_test.append(data_wo_labels.values)

    specs = {}
    specs['epochs'] = 1
    specs['features'] = 40
    train_model(x_train, y_train, specs)
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

import os
import pandas as pd

cur_dir = os.getcwd()
folder = cur_dir + "/../data/sepsis_data/"
train_labels = []
test_labels = []
dir = cur_dir + "/../sepsis_data_txt/"
if not os.path.exists(dir):
    os.makedirs(dir)

train_dir = dir + "train/"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

test_dir = dir + "test/"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_test_split = 0

for subdir, dirs, files in os.walk(folder):
    for file in files:
        to_save_lst = []
        fullFile = os.path.join(subdir, file)
        preprocessed = []

        split_by_line = open(fullFile).read().split('\n')

        for line in split_by_line:
            preprocessed.append(line.split('|'))

        df = pd.DataFrame(preprocessed[1:-1], columns=preprocessed[0])

        label = df['SepsisLabel'].tolist()[-1]

        if train_test_split % 4 == 0:
            test_labels.append([file[0:-4], label])
            dir_to_save = test_dir
        else:
            train_labels.append([file[0:-4], label])
            dir_to_save = train_dir

        demographic_features = list(df)[-7:-1]
        other_features = list(df)[0:-7]

        time = 0
        time_to_save = "00:00"

        for demographic_feature in demographic_features:
            to_save_lst.append([time_to_save, demographic_feature, df[demographic_feature].tolist()[0]])

        df.drop(demographic_features + ['SepsisLabel'], axis=1, inplace=True)

        iter = df.iterrows()

        for row in iter:
            time += 1

            if time < 10:
                time_to_save = "0" + str(time) + ":00"
            else:
                time_to_save = str(time) + ":00"

            for other_feature in other_features:
                if row[1][other_feature] != 'NaN':
                    to_save_lst.append([time_to_save, other_feature, row[1][other_feature]])

        train_test_split += 1

        pd.DataFrame(to_save_lst, columns=['Time', 'Parameter',
            'Value']).to_csv('{}{}.txt'.format(dir_to_save,file[0:-4]), index=False)

    pd.DataFrame(train_labels, columns=['PID', 'SepsisLabel']).to_csv('{}list.txt'.format(train_dir), index=False)
    pd.DataFrame(test_labels, columns=['PID', 'SepsisLabel']).to_csv('{}list.txt'.format(test_dir), index=False)

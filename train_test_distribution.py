import os
import pandas as pd

cur_dir = os.getcwd()
train_list = cur_dir + "/sepsis_data_txt/train/list.txt"
test_list = cur_dir + "/sepsis_data_txt/test/list.txt"


train_df = pd.read_csv(train_list)
test_df = pd.read_csv(test_list)

print("Training Set")
print(train_df['SepsisLabel'].value_counts())
print("Testing Set")
print(test_df['SepsisLabel'].value_counts())

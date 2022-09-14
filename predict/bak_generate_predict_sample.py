import pandas as pd
import time
import datetime
import numpy as np

import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

type = 'pred'

if __name__ == '__main__':
    if type == 'pred':
        delta = 3
        cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
        cur_time = '20220725'
        sample_predict_path = '../data/sample_predict_{}.csv'.format(cur_time)
        raw_sample_file = '../data/raw_sample_{}.csv'.format(cur_time)
        raw_sample = pd.read_csv(raw_sample_file)
        print(raw_sample.describe())
        data_pred = raw_sample.head(10000)
        data_pred.to_csv(sample_predict_path)
    else:
        delta = 3
        cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
        cur_time = '20220725'
        raw_sample_file = '../data/raw_sample_{}.csv'.format(cur_time)
        raw_sample = pd.read_csv(raw_sample_file)
        print(raw_sample.describe())
        data_train = raw_sample.head(10000)
        data_train_file = '../data/raw_sample_test.csv'
        data_train.to_csv(data_train_file)

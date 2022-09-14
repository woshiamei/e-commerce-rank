import pandas as pd
import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

import argparse
"""
多目标建模
"""

parse = argparse.ArgumentParser()
parse.add_argument('--date', type=str, default='20220802')
# parse.add_argument('--retrain', type=str, default='true')
args = parse.parse_args()

pred_date = datetime.datetime.strptime(args.date, '%Y%m%d').strftime("%Y%m%d")

delta = datetime.timedelta(days=1)

batch_size = 64

def full_predict(data_path, cur_time, model_dir):
    print('data_path {}'.format(data_path))
    print('cur time {}'.format(cur_time))
    print('model dir {}'.format(model_dir))
    data = pd.read_csv(data_path)
    print(data.describe())
    # 增加一列 action
    # raw_sample['action'] = raw_sample['p_click'] + raw_sample['is_addtocart'] + raw_sample['is_order']
    # raw_sample['acvr'] = raw_sample['is_addtocart'] + raw_sample['is_order']

    # 采样： 增加一列 random , 根据 view action
    # raw_sample["random"] = raw_sample.apply(lambda x: np.random.rand(), axis=1)
    # data = raw_sample[raw_sample["action"] + raw_sample["random"] >= 0.5]

    # 小数据测试
    # data.rename(columns={"action": "target"})
    # data = data.head(10000)

    # xtr * 1e3
    mean = data["ctr"].mean()
    data["ctr"] = data.apply(lambda x: x.ctr * 1e2 if x.ctr is not None else mean * 1e2, axis=1)
    mean = data["atr"].mean()
    data["atr"] = data.apply(lambda x: x.atr * 1e2 if x.atr is not None else mean * 1e2, axis=1)
    mean = data["cvr"].mean()
    data["cvr"] = data.apply(lambda x: x.cvr * 1e2 if x.cvr is not None else mean * 1e2, axis=1)

    dict1 = {'order_first_time': '2000-01-01 00:00:00', 'order_latest_time': '2000-01-01 00:00:00'}
    data.fillna(dict1, inplace=True)
    data.order_latest_time = data.apply(lambda x : (datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_latest_time, '%Y-%m-%d %H:%M:%S')).days, axis=1)
    data.order_first_time = data.apply(lambda x : (datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_first_time, '%Y-%m-%d %H:%M:%S')).days, axis=1)

    data = data.apply(lambda x: x.fillna(-1) if x.dtype.kind in 'biufc' else x.fillna('-'))
    for fea in ['orders_count', 'orders_month', 'orders_3month', 'orders_6month', 'orders_year', 'discount_num', 'comment_num', 'show_sum', 'click_sum', 'add_sum', 'order_sum']:
        data[fea] = data[fea].astype(int)
    for fea in ['fullvisitorid', 'productsku', 'id', 'gender', 'lace_size', 'pro_type', 'product_color', 'product_texture', 'wigtype']:
        if data[fea].dtype.kind in 'f':
            data[fea] = data[fea].astype(int)
        data[fea] = data[fea].astype(str)

    print('data examples num is {}'.format(len(data)))

    t2 = time.time()

    # 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
    def df_to_dataset(dataframe, shuffle=False, batch_size=32):
        dataframe = dataframe.copy()
        # df_labels = dataframe[['target_1', 'target_2']]
        # ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), (list(df_labels.target_1), list(df_labels.target_2))))
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    data_ds = df_to_dataset(data, batch_size=batch_size)

    print("load unice_rank_model {}..........".format(model_dir))
    model = tf.keras.models.load_model(model_dir)

    # loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(data_ds)
    # print("pctr_accuracy", pctr_accuracy)
    # print("pcvr_accuracy", pcvr_accuracy)
    res = model.predict(data_ds)
    print("res: {}".format(res))
    # 转 df
    pred_res_df = pd.DataFrame()
    pred_res_df[['vid', 'uid', 'pid']] = data[['fullvisitorid', 'id', 'productsku']]
    pred_res_df[['pctr', 'pcvr']] = pd.DataFrame(np.concatenate([res[0], res[1]],axis=1))
    # 将 多个label 拼在一起
    pred_res_df.columns = ['vid', 'uid', 'pid', 'pctr', 'pcvr']

    print("pred_res_df: ")
    # print(pred_res_df.describe())
    pred_res_file = '../pred_data/pred_res_{}.csv'.format(cur_time)
    pred_res_df.to_csv(pred_res_file, index=False)

    t3 = time.time()
    print("全量预估执行总花费时间： {}".format(t3 - t1))
    print("数据预处理花费时间： {}".format(t2 - t1))
    print("模型全量预估花费时间： {}".format(t3 - t2))

def get_model_path(model_time):
    # 前五天
    deltas = 5
    model_dir = ""
    for i in range(1, deltas):
        model_time = (datetime.datetime.strptime(model_time, '%Y%m%d') - datetime.timedelta(days=i)).strftime("%Y%m%d")
        # 先以 v2 版本为主
        model_dir = "../model/unice_rank_model_v2_{}".format(model_time)
        if os.path.exists(model_dir):
            break
        else:
            print("model path not exist {}".format(model_dir))
    if not os.path.exists(model_dir):
        print("************************************ retrain failed *************************")
        print("retrain failed, model not exist!")
        exit()
    return model_dir

if __name__ == '__main__':
    print("start predict ....")
    t1 = time.time()

    data_path = '../pred_data/sample_predict_{}.csv'.format(pred_date)
    if os.path.exists(data_path):
        print('data_path exits')
        model_path = get_model_path(pred_date)
        full_predict(data_path, pred_date, model_path)
    else:
        print('data_path {} not exits'.format(data_path))

    t2 = time.time()
    print("全量预估执行总花费时间： {}".format(t2 - t1))


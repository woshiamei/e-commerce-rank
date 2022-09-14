import distributed.scheduler
import pandas as pd
import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

import sys

sys.path.append("/root/reco/rank/train")
from feature_handle_v3 import features_extract, fea_inputs

import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

import argparse
"""
多目标建模
"""
parse = argparse.ArgumentParser()
parse.add_argument('--start_date', type=str, default='20220715')
# parse.add_argument('--end_date', type=str, default='20221001')
parse.add_argument('--date_num', type=int, default=30)
parse.add_argument('--overwrite', type=bool, default=False)
parse.add_argument('--retrain', type=bool, default=True)
# 正常训练记得关闭
parse.add_argument('--allow_cold_start', type=bool, default=True)
args = parse.parse_args()


start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
# end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')
date_num = args.date_num
# overwrite = args.overwrite
# retrain = args.retrain

model_path_prefix = 'unice_rank_model_v6_2'
batch_size = 128
delta = datetime.timedelta(days=1)

def train_process(data_path, cur_time, model_dir):
    print("data handle cur_time: {}".format(cur_time))
    # raw_sample_file = "../data/sample_join.csv"
    t1 = time.time()

    model_time = (datetime.datetime.strptime(cur_time, '%Y%m%d') - delta).strftime("%Y%m%d")
    # print('model path {}'.format(model_dir))
    raw_sample_file = data_path

    # , dtype={'wigtype':str, 'product_texture':str, 'product_color':str, 'pro_type':str}
    raw_sample = pd.read_csv(raw_sample_file)

    # for i in ['wigtype', 'product_texture', 'product_color', 'pro_type', 'lace_size']:
    #     raw_sample[i] = pd.Categorical(raw_sample[i])
    #     raw_sample[i] = raw_sample[i].cat.codes

    print(raw_sample.describe())

    # 过滤：
    # show num 小于 1000 ， xtr 为空
    raw_sample = raw_sample[raw_sample.show_sum > 1000]
    raw_sample.dropna(subset=['ctr', 'atr', 'cvr'], inplace=True)
    # for xtr in ['ctr', 'atr', 'cvr']:
    # 	data[xtr] = data[xtr].fillna(0.0)

    # 增加一列 action
    raw_sample['action'] = raw_sample['p_click'] + raw_sample['is_addtocart'] + raw_sample['is_order']
    raw_sample['acvr'] = raw_sample['is_addtocart'] + raw_sample['is_order']

    # 采样： 增加一列 random , 根据 view action
    raw_sample["random"] = raw_sample.apply(lambda x: np.random.rand(), axis=1)
    data = raw_sample[raw_sample["action"] + raw_sample["random"] >= 0.5]

    # 小数据测试
    # data.rename(columns={"action": "target"})
    # data = data.head(10000)

    # xtr 使用均值进行填充
    mean = data["ctr"].mean()
    data['ctr'] = data['ctr'].fillna(mean)
    data["ctr"] = data.apply(lambda x: x.ctr * 1e2, axis=1)
    mean = data["atr"].mean()
    data['atr'] = data['atr'].fillna(mean)
    data["atr"] = data.apply(lambda x: x.atr * 1e2, axis=1)
    # data["atr"] = data.apply(lambda x: x.atr * 1e2 if x.atr is not None else mean * 1e2, axis=1)
    mean = data["cvr"].mean()
    data['cvr'] = data['cvr'].fillna(mean)
    data["cvr"] = data.apply(lambda x: x.cvr * 1e2, axis=1)

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

    data['target_1'] = np.where(data['p_click'] > 0, 1, 0)
    data['target_2'] = np.where(data['acvr'] > 0, 1, 0)
    data = data.drop(columns=['action', 'random', 'p_view', 'p_click', 'is_addtocart', 'is_order', 'email', 'date', 'acvr'])

    # train test
    # train_all, test_no = train_test_split(data, test_size=0.001)
    train, test = train_test_split(data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    t2 = time.time()
    # print("sql 查询耗时： {}".format(t2 - t1))

    # 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        df_labels = dataframe[['target_1', 'target_2']]
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), (list(df_labels.target_1), list(df_labels.target_2))))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    # train_ds = df_to_dataset(train, batch_size=batch_size)
    train_ds = df_to_dataset(data, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    if not os.path.exists(model_dir):
        print("model construct ..........")

        # model
        feature_inputs = fea_inputs()
        feature_columns = features_extract()
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        inputs = feature_layer(feature_inputs)
        # inputs = tf.keras.layers.DenseFeatures(feature_columns)(feature_inputs)

        output = tf.keras.layers.Dense(256, name='bottom_1', activation=tf.nn.relu)(inputs)
        output_ctr = tf.keras.layers.Dense(128, name='ctr_1', activation=tf.nn.relu)(output)
        pctr = tf.keras.layers.Dense(1, name='pctr', activation=tf.nn.sigmoid)(output_ctr)

        output_cvr = tf.keras.layers.Dense(256, name='bottom_2', activation=tf.nn.relu)(inputs)
        output_cvr = tf.keras.layers.Dense(128, name='cvr_1', activation=tf.nn.relu)(output_cvr)
        pcvr = tf.keras.layers.Dense(1, name='pcvr', activation=tf.nn.sigmoid)(output_cvr)
        losses = {
            'pctr': 'binary_crossentropy',
            'pcvr': 'binary_crossentropy'
        }
        loss_weights={
            'pctr': 1.,
            'pcvr': 1.
        }
        model = tf.keras.Model(inputs=feature_inputs, outputs=[pctr, pcvr])
        model.compile(optimizer='adam',
                    loss=losses,
                    loss_weights=loss_weights,
                    weighted_metrics=[],
                    metrics=['accuracy'],
                    run_eagerly=True)
    else:
        print("load unice_rank_model {}..........".format(model_dir))
        model = tf.keras.models.load_model(model_dir)

    # tensorboard --logdir log
    # PS D:\gaolei\rank\train> tensorboard --logdir log
    #linux
    # ssh -L 16006:127.0.0.1:6006 root@192.168.1.150
    # tensorboard --logdir=log --port=60
    # 127.0.0.1:16006

    log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_ds,
              validation_data=val_ds,
              epochs=2,
              verbose=1,
              callbacks=[tensorboard_callback]
                        )

    loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(test_ds)
    print("pctr_accuracy", pctr_accuracy)
    print("pcvr_accuracy", pcvr_accuracy)

    print(model.predict(test_ds))

    model_dir_new = "../model/{}_{}".format(model_path_prefix, cur_time)
    model.save(model_dir_new)

    keras.backend.clear_session()

    t3 = time.time()
    print("当前训练的日期数据： {}".format(cur_time))
    print("执行总花费时间： {}".format(t3 - t1))
    print("数据预处理花费时间： {}".format(t2 - t1))
    print("模型训练花费时间： {}".format(t3 - t2))

if __name__ == '__main__':
    print("start train ....")
    print('start_date: {},  date_num: {}'.format(args.start_date, args.date_num))
    t1 = time.time()
    for i in range(0, date_num):
        date = (start_date + datetime.timedelta(days=i)).strftime("%Y%m%d")
        print('data date {}'.format(date))
        data_path = '../data/raw_sample_{}.csv'.format(date)
        cur_model_path = '../model/{}_{}'.format(model_path_prefix, date)
        if os.path.exists(data_path):
            if os.path.exists(cur_model_path):
                print("modelapth {} exist!".format(cur_model_path))
                if not args.overwrite:
                    print("Don't overwrite!")
                    continue
            else:
                model_time = date
                model_dir = "../model/{}_{}".format(model_path_prefix, model_time)
                if args.retrain:
                    # 前五天
                    deltas = 5
                    for i in range(1, deltas):
                        model_time = (datetime.datetime.strptime(model_time, '%Y%m%d') - datetime.timedelta(days=i)).strftime("%Y%m%d")
                        model_dir = "../model/{}_{}".format(model_path_prefix, model_time)
                        if os.path.exists(model_dir):
                            break
                        else:
                            print("model path not exist {}".format(model_dir))
                    if not os.path.exists(model_dir):
                        print("************************************ retrain failed *************************")
                        print("retrain failed, model not exist!")
                        if not args.allow_cold_start:
                            print("******************** not allow clod start mode ***********************")
                            exit()
                        print("************************************ cold start mode *************************")
            train_process(data_path, date, model_dir)
        else:
            print("data path not exist {}".format(data_path))

    t2 = time.time()
    print('start_date: {},  date_num: {}'.format(args.start_date, args.date_num))
    print("执行总花费时间： {}".format(t2 - t1))


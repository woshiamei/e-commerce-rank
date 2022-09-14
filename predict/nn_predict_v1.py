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

"""
多目标 预估
"""

delta = 3
cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
cur_time = '20220725'

model_time = (datetime.datetime.now() - datetime.timedelta(days=delta + 1)).strftime("%Y-%m-%d").replace("-", "")
model_time = '20220716'

print("load model time: {}".format(model_time))
print("handle predict data, cur_time: {}".format(cur_time))
t1 = time.time()

raw_sample_file = "../data/sample_predict_{}.csv".format(cur_time)

# , dtype={'wigtype':str, 'product_texture':str, 'product_color':str, 'pro_type':str}
raw_sample = pd.read_csv(raw_sample_file)

for i in ['wigtype', 'product_texture', 'product_color', 'pro_type', 'lace_size']:
  raw_sample[i] = pd.Categorical(raw_sample[i])
  raw_sample[i] = raw_sample[i].cat.codes

print(raw_sample.head())
# 增加一列 action
raw_sample['action'] = raw_sample['p_click'] + raw_sample['is_addtocart'] + raw_sample['is_order']
raw_sample['acvr'] = raw_sample['is_addtocart'] + raw_sample['is_order']

# 采样： 增加一列 random , 根据 view action
raw_sample["random"] = raw_sample.apply(lambda x: np.random.rand(), axis=1)
data = raw_sample[raw_sample["action"] + raw_sample["random"] >= 0.5]

# 小数据测试
# data.rename(columns={"action": "target"})
data = data.head(10000)

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

data = data.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
for fea in ['orders_count', 'orders_month', 'orders_3month', 'orders_6month', 'orders_year', 'discount_num', 'comment_num', 'show_sum', 'click_sum', 'add_sum', 'order_sum']:
  data[fea] = data[fea].astype(int)
for fea in ['fullvisitorid', 'productsku', 'id', 'lace_size', 'pro_type', 'product_color', 'product_texture', 'wigtype']:
  data[fea] = data[fea].astype(str)

# d = data[data.order_first_time != 8232]
# d = data[data.order_first_time != 8232]
# d['order_latest_time'].quantile(0.99)

data['target_1'] = np.where(data['p_click'] > 0, 1, 0)
data['target_2'] = np.where(data['acvr'] > 0, 1, 0)
data = data.drop(columns=['action', 'random', 'p_view', 'p_click', 'is_addtocart', 'is_order', 'email', 'date', 'acvr'])


t2 = time.time()

# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  df_labels = dataframe[['target_1', 'target_2']]
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), (list(df_labels.target_1), list(df_labels.target_2))))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 128
data_ds = df_to_dataset(data, batch_size=batch_size)

# model
# features_extract()
# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# fea_inputs()
# inputs = feature_layer(feature_inputs)


model_dir = "../model/unice_rank_model_v1_{}".format(model_time)
if not os.path.exists(model_dir):
  print("model not exits, check model dir {}!".format(model_dir))
  # model = tf.keras.Sequential([
  #   feature_layer,
  #   layers.Dense(128, activation='relu'),
  #   layers.Dense(128, activation='relu'),
  #   layers.Dense(1, activation='sigmoid')
  # ])
#   output = tf.keras.layers.Dense(256, name='bottom_1', activation=tf.nn.relu)(inputs)
#   output_ctr = tf.keras.layers.Dense(128, name='ctr_1', activation=tf.nn.relu)(output)
#   pctr = tf.keras.layers.Dense(1, name='pctr', activation=tf.nn.sigmoid)(output_ctr)
#
#   output_cvr = tf.keras.layers.Dense(256, name='bottom_2', activation=tf.nn.relu)(inputs)
#   output_cvr = tf.keras.layers.Dense(128, name='cvr_1', activation=tf.nn.relu)(output_cvr)
#   pcvr = tf.keras.layers.Dense(1, name='pcvr', activation=tf.nn.sigmoid)(output_cvr)
#   losses = {
#     'pctr': 'binary_crossentropy',
#     'pcvr': 'binary_crossentropy'
#   }
#   loss_weights={
#     'pctr': 1.,
#     'pcvr': 1.
#   }
#   model = tf.keras.Model(inputs=feature_inputs, outputs=[pctr, pcvr])
#   model.compile(optimizer='adam',
#                 loss=losses,
#                 loss_weights=loss_weights,
#                 weighted_metrics=[],
#                 metrics=['accuracy'],
#                 run_eagerly=True)
else:
  print("load unice_rank_model dir : {}..........".format(model_dir))
  model = tf.keras.models.load_model(model_dir)
#
# history = model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=2,
#           verbose=1)

# model.summary()

# loss, accuracy = model.evaluate(test_ds)
# print("accuracy: ", accuracy)

  loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(data_ds)
  print("pctr_accuracy", pctr_accuracy)
  print("pcvr_accuracy", pcvr_accuracy)

  print(model.predict(data_ds))
# 转 df
# d = pd.DataFrame({'pctr': list(res[0]), 'pcvr': list(res[1])})


t3 = time.time()
print("执行总花费时间： {}".format(t3 - t1))
print("数据预处理花费时间： {}".format(t2 - t1))
print("模型训练花费时间： {}".format(t3 - t2))
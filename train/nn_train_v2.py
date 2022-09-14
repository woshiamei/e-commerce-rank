import pandas as pd
import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

import sys

sys.path.append("/root/reco/rank/train")
from feature_handle import features_extract, fea_inputs

import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

"""
多目标建模
"""

delta = 3
cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
cur_time = '20220716'
print("data handle cur_time: {}".format(cur_time))
# raw_sample_file = "../data/sample_join.csv"
t1 = time.time()

raw_sample_file = "../data/raw_sample_{}.csv".format(cur_time)
# raw_sample_file = '../data/raw_sample_test.csv'

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
# data = data.head(10000)

# xtr * 1e3
mean = data["ctr"].mean()
data["ctr"] = data.apply(lambda x: x.ctr * 1e2 if x.ctr is not None else mean * 1e2, axis=1)
mean = data["atr"].mean()
data["atr"] = data.apply(lambda x: x.atr * 1e2 if x.atr is not None else mean * 1e2, axis=1)
mean = data["cvr"].mean()
data["cvr"] = data.apply(lambda x: x.cvr * 1e2 if x.cvr is not None else mean * 1e2, axis=1)
# for xtr in ["ctr", "atr", "cvr"]:
#   mean = data[xtr].mean()
#   data[xtr] = data.apply(lambda x: x.xtr * 1e3 if x.xtr is not None else mean * 1e3, axis=1)

# d = data[data["order_first_time"].notnull()]
# d.order_first_time
# raw_sample.head()
# d = raw_sample[raw_sample["order_first_time"].notnull()]
# print(d.email)
# d2 = d.order_first_time[:1]
# d3 = datetime.datetime.strptime(d2.values[0], '%Y-%m-%d %H:%M:%S')
# d4 = datetime.datetime.strptime(cur_time, '%Y%m%d')
# delta = d4 - d3

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

# data['target_2'].sum()
# Out[1]: 2477
# data['target_1'].sum()
# Out[2]: 13110
# data.shape
# Out[3]: (115826, 53)

# train test
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

t2 = time.time()
# print("sql 查询耗时： {}".format(t2 - t1))

# batch_size = 64
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
#
#
# for feature_batch, label_batch in train_ds.take(1):
#   print('Every feature:', list(feature_batch.keys()))
#   print('A batch of maxtotal:', feature_batch['maxtotal'])
#   print('A batch of targets:', label_batch)
#
# example_batch = next(iter(train_ds))[0]

# def demo(feature_column):
#   feature_layer = layers.DenseFeatures(feature_column)
#   print(feature_layer(example_batch).numpy())
#
# ctr = feature_column.numeric_column("ctr")
# demo(ctr)
#
# show_num = feature_column.numeric_column("show_num")

# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  df_labels = dataframe[['target_1', 'target_2']]
  # labels_ctr = dataframe.pop('target_1')
  # labels_cvr = dataframe.pop('target_2')
  # labels = dataframe["target_1"]
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), (list(df_labels.target_1), list(df_labels.target_2))))
  # for element in ds.as_numpy_iterator():
  #   print(element)
  #   break
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 512
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# model
feature_inputs = fea_inputs()
feature_columns = features_extract()
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
inputs = feature_layer(feature_inputs)
# inputs = tf.keras.layers.DenseFeatures(feature_columns)(feature_inputs)

model_dir = "../model/unice_rank_model_v1_{}".format(cur_time)
if not os.path.exists(model_dir):
  print("model construct ..........")
  # model = tf.keras.Sequential([
  #   feature_layer,
  #   layers.Dense(128, activation='relu'),
  #   layers.Dense(128, activation='relu'),
  #   layers.Dense(1, activation='sigmoid')
  # ])
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
  print("load unice_rank_model {}..........".format(cur_time))
  model = tf.keras.models.load_model(model_dir)


# summary
# tensorboard --logdir log
# PS D:\gaolei\rank\train> tensorboard --logdir log
log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=2,
          verbose=1,
          callbacks=[tensorboard_callback]
                    )


# import matplotlib.pyplot as plt
#
# def plot_metric(history, metric):
#   train_metrics = history.history[metric]
#   val_metrics = history.history['val_' + metric]
#   epochs = range(1, len(train_metrics) + 1)
#   plt.plot(epochs, train_metrics, 'bo--')
#   plt.plot(epochs, val_metrics, 'ro-')
#   plt.title('Training and validation ' + metric)
#   plt.xlabel("Epochs")
#   plt.ylabel(metric)
#   plt.legend(["train_" + metric, 'val_' + metric])
#   plt.show()
#
#
# plot_metric(history, "accuracy")

# model.summary()

# loss, accuracy = model.evaluate(test_ds)
# print("accuracy: ", accuracy)
loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(test_ds)
print("pctr_accuracy", pctr_accuracy)
print("pcvr_accuracy", pcvr_accuracy)

print(model.predict(test_ds))

model.save(model_dir)


t3 = time.time()
print("执行总花费时间： {}".format(t3 - t1))
print("数据预处理花费时间： {}".format(t2 - t1))
print("模型训练花费时间： {}".format(t3 - t2))
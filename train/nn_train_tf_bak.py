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
多目标建模
"""

delta = 3
cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
cur_time = '20220716'
print("data handle cur_time: {}".format(cur_time))
# raw_sample_file = "../data/sample_join.csv"
t1 = time.time()

raw_sample_file = "../data/raw_sample_{}.csv".format(cur_time)

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
# train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
# print(len(val), 'validation examples')
print(len(test), 'test examples')

t2 = time.time()

#
# show_num = feature_column.numeric_column("show_num")
#
# show_num_buckets = feature_column.bucketized_column(show_num, boundaries=[5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000, 3000000])
# demo(show_num_buckets)

# Index(['fullvisitorid', 'productsku', 'email', 'id', 'date', 'p_view',
#        'p_click', 'is_addtocart', 'is_order', 'gender', 'age_range', 'country',
#        'region', 'city', 'register_channel', 'register_channel_type',
#        'register_type_one', 'register_type_two', 'orders_count', 'totalspent',
#        'maxtotal', 'atv_range', 'order_first_time', 'order_latest_time',
#        'orders_month', 'orders_3month', 'orders_6month', 'orders_year',
#        'discount_num', 'discount_price', 'pay_method', 'comment',
#        'comment_num', 'reward_point', 'use_reward_point', 'level',
#        'order_range', 'r', 'f', 'm', 'newcustomer', 'ordercustomer',
#        'datediff_day', 'customer_type', 'customer_type_finance', 'pro_type',
#        'product_color', 'product_texture', 'lace_size', 'wigtype', 'show_sum',
#        'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr', 'action'],)

feature_columns = []

# id 类列
uid_hashed = feature_column.categorical_column_with_hash_bucket('id', hash_bucket_size=100000)
vid_hashed = feature_column.categorical_column_with_hash_bucket('fullvisitorid', hash_bucket_size=1000000)
pid_hashed = feature_column.categorical_column_with_hash_bucket('productsku', hash_bucket_size=10000)

# 嵌入列 : id
uid_embedding = feature_column.embedding_column(uid_hashed, dimension=16)
vid_embedding = feature_column.embedding_column(vid_hashed, dimension=32)
pid_embedding = feature_column.embedding_column(pid_hashed, dimension=16)
feature_columns.extend([uid_embedding, vid_embedding, pid_embedding])

cate_list = ['gender', 'age_range', 'country', 'region', 'city', 'register_channel', 'register_channel_type', 'register_type_one', 'register_type_two']
for fea in cate_list:
  fea_hash = feature_column.categorical_column_with_hash_bucket(fea, hash_bucket_size=50)
  feature_columns.append(feature_column.embedding_column(fea_hash, dimension=4))

# 分桶列
# order_count = feature_column.numeric_column('orders_count')
# order_count_buckets = feature_column.bucketized_column(order_count, boundaries=[i for i in range(8)])
# feature_columns.append(order_count_buckets)

# order_list = ['totalspent', 'maxtotal', 'atv_range', 'order_first_time', 'order_latest_time']
total_spent_count = feature_column.numeric_column('totalspent')
total_spent_buckets = feature_column.bucketized_column(total_spent_count, boundaries=[50, 98, 120 ,180 , 240, 302, 510, 743, 1100, 1439])
feature_columns.append(total_spent_buckets)

max_total_count = feature_column.numeric_column('maxtotal')
max_total_buckets = feature_column.bucketized_column(max_total_count, boundaries=[50, 95, 120, 164, 200, 236, 300, 366, 450, 535])
feature_columns.append(max_total_buckets)

atv_range_hashed = feature_column.categorical_column_with_hash_bucket('atv_range', hash_bucket_size=10)
feature_columns.append(feature_column.embedding_column(atv_range_hashed, dimension=4))

# order_first_time order_latest_time 求解 diff

order_first_time = feature_column.numeric_column('order_first_time')
order_first_time_buckets = feature_column.bucketized_column(order_first_time, boundaries=[-1, 0, 15, 60, 120, 200, 300, 480, 790, 1927])
feature_columns.append(order_first_time_buckets)

order_latest_time = feature_column.numeric_column('order_latest_time')
order_latest_time_buckets = feature_column.bucketized_column(order_latest_time, boundaries=[-1, 0, 9, 30, 50, 70, 120, 250, 600, 1070])
feature_columns.append(order_latest_time_buckets)

# 'orders_month', 'orders_3month', 'orders_6month', 'orders_year',
orders_list = ['orders_count', 'orders_month', 'orders_3month', 'orders_6month', 'orders_year', 'discount_num']
for fea in orders_list:
  fea = feature_column.numeric_column(fea)
  fea_buckets = feature_column.bucketized_column(fea, boundaries=[i for i in range(8)])
  feature_columns.append(fea_buckets)

discount_price = feature_column.numeric_column('discount_price')
discount_price_buckets = feature_column.bucketized_column(discount_price, boundaries=[1, 4, 18, 55, 117])
feature_columns.append(discount_price_buckets)

pay_method_hashed = feature_column.categorical_column_with_hash_bucket('pay_method', hash_bucket_size=20)
pay_method_embedding = feature_column.embedding_column(pay_method_hashed, dimension=4)
feature_columns.append(pay_method_embedding)

comment_hashed = feature_column.categorical_column_with_hash_bucket('comment', hash_bucket_size=2)
feature_columns.append(feature_column.indicator_column(comment_hashed))

comment_num = feature_column.numeric_column('comment_num')
comment_num_hashed = feature_column.bucketized_column(comment_num, boundaries=[i for i in range(10)])
feature_columns.append(feature_column.embedding_column(comment_num_hashed, dimension=4))

reward_point = feature_column.numeric_column('reward_point')
reward_point_buckets = feature_column.bucketized_column(reward_point, boundaries=[20, 49, 99, 198, 500, 1000, 2715, 7406])
feature_columns.append(reward_point_buckets)

use_reward_point = feature_column.numeric_column('use_reward_point')
use_reward_point_buckets = feature_column.bucketized_column(use_reward_point, boundaries=[0, 25, 50, 100, 109, 250, 500, 700])
feature_columns.append(use_reward_point_buckets)

level_hashed = feature_column.categorical_column_with_hash_bucket('level', hash_bucket_size=10)
feature_columns.append(feature_column.embedding_column(level_hashed, dimension=4))

custom_list = ['order_range', 'r', 'f', 'm', 'newcustomer', 'ordercustomer']
for custom in custom_list:
  custom_hashed = feature_column.categorical_column_with_hash_bucket(custom, hash_bucket_size=10)
  feature_columns.append(feature_column.embedding_column(custom_hashed, dimension=4))

datediff_day_buckets = feature_column.categorical_column_with_hash_bucket('datediff_day', hash_bucket_size=10)
feature_columns.append(feature_column.indicator_column(datediff_day_buckets))

for custom_type in ['customer_type', 'customer_type_finance']:
  custom_type_hashed = feature_column.categorical_column_with_hash_bucket(custom_type, hash_bucket_size=4)
  feature_columns.append(feature_column.indicator_column(custom_type_hashed))

# item
feature_list = ['pro_type', 'product_color', 'product_texture', 'lace_size', 'wigtype']
for fea in feature_list:
  fea_hashed = feature_column.categorical_column_with_hash_bucket(fea, hash_bucket_size=10)
  feature_columns.append(feature_column.embedding_column(fea_hashed, dimension=4))

#        'show_sum', 'click_sum', 'add_sum', 'order_sum',

# show_sum
# [12500.5,41498.0,182952.0,681307.8,1905942.6399999973]
# click_sum
# [371.0,1501.0,6901.0,43113.49999999992,95173.83999999987]
# add_sum
# [39.0,197.0,1053.5,6878.299999999999,17992.279999999966]
# order_sum
# [4.0,26.0,156.0,956.5999999999995,3531.539999999988]
# ctr
# [0.02242302242666483,0.04226813092827797,0.06659471616148949,0.10877974554896354,0.1723250064253805]
# atr
# [0.0013332038070075214,0.004906301386654377,0.010905020404607058,0.03015259448438882,0.15449105888605008]
# cvr
# [1.5758725930936635E-4,5.792623851448298E-4,0.0013314966927282512,0.004849342443048952,0.16571623235940874]
show_num_list = [2e3, 8e3, 1.25e4, 2.5e4, 4.15e4, 1e5, 1.8e5, 3e5, 6.8e5, 1.8e6]
fea = feature_column.numeric_column('show_sum')
fea_buckets = feature_column.bucketized_column(fea, boundaries=show_num_list)
feature_columns.append(fea_buckets)

click_num_list = [100, 310, 800, 1500, 4000, 6900, 20000, 43000, 60000, 95000]
fea = feature_column.numeric_column('click_sum')
fea_buckets = feature_column.bucketized_column(fea, boundaries=click_num_list)
feature_columns.append(fea_buckets)

add_num_list = [10, 39, 120, 197, 600, 1053, 4000, 6878, 12000, 18000]
fea = feature_column.numeric_column('add_sum')
fea_buckets = feature_column.bucketized_column(fea, boundaries=add_num_list)
feature_columns.append(fea_buckets)

order_num_list = [1, 4, 15, 26, 90, 156, 550, 956, 2200, 3500]
fea = feature_column.numeric_column('order_sum')
fea_buckets = feature_column.bucketized_column(fea, boundaries=order_num_list)
feature_columns.append(fea_buckets)
# for fea in ['show_sum', 'click_sum', 'add_sum', 'order_sum']:
#   fea = feature_column.numeric_column(fea)
#   fea_buckets = feature_column.bucketized_column(fea, boundaries=show_num_list)
#   feature_columns.append(fea_buckets)

# 'ctr', 'atr', 'cvr',
# 数值列
# 假设 xtr 乘以 1000
for fea in ['ctr', 'atr', 'cvr']:
  feature_columns.append(feature_column.numeric_column(fea))

feature_inputs = {}
feature_inputs['fullvisitorid'] = tf.keras.Input((1,), name='fullvisitorid', dtype=tf.string)
feature_inputs['productsku'] = tf.keras.Input((1,), name='productsku', dtype=tf.string)
# feature_inputs['email'] = tf.keras.Input((1,), name='email', dtype=tf.string)
feature_inputs['id'] = tf.keras.Input((1,), name='id', dtype=tf.string)
# feature_inputs['date'] = tf.keras.Input((1,), name='date', dtype=tf.int64)
feature_inputs['gender'] = tf.keras.Input((1,), name='gender', dtype=tf.string)
feature_inputs['age_range'] = tf.keras.Input((1,), name='age_range', dtype=tf.string)
feature_inputs['country'] = tf.keras.Input((1,), name='country', dtype=tf.string)
feature_inputs['region'] = tf.keras.Input((1,), name='region', dtype=tf.string)
feature_inputs['city'] = tf.keras.Input((1,), name='city', dtype=tf.string)
feature_inputs['register_channel'] = tf.keras.Input((1,), name='register_channel', dtype=tf.string)
feature_inputs['register_channel_type'] = tf.keras.Input((1,), name='register_channel_type', dtype=tf.string)
feature_inputs['register_type_one'] = tf.keras.Input((1,), name='register_type_one', dtype=tf.string)
feature_inputs['register_type_two'] = tf.keras.Input((1,), name='register_type_two', dtype=tf.string)
feature_inputs['orders_count'] = tf.keras.Input((1,), name='orders_count', dtype=tf.int32)
feature_inputs['totalspent'] = tf.keras.Input((1,), name='totalspent', dtype=tf.float64)
feature_inputs['maxtotal'] = tf.keras.Input((1,), name='maxtotal', dtype=tf.float64)
feature_inputs['atv_range'] = tf.keras.Input((1,), name='atv_range', dtype=tf.string)
feature_inputs['order_first_time'] = tf.keras.Input((1,), name='order_first_time', dtype=tf.int64)
feature_inputs['order_latest_time'] = tf.keras.Input((1,), name='order_latest_time', dtype=tf.int64)
feature_inputs['orders_month'] = tf.keras.Input((1,), name='orders_month', dtype=tf.int32)
feature_inputs['orders_3month'] = tf.keras.Input((1,), name='orders_3month', dtype=tf.int32)
feature_inputs['orders_6month'] = tf.keras.Input((1,), name='orders_6month', dtype=tf.int32)
feature_inputs['orders_year'] = tf.keras.Input((1,), name='orders_year', dtype=tf.int32)
feature_inputs['discount_num'] = tf.keras.Input((1,), name='discount_num', dtype=tf.int32)
feature_inputs['discount_price'] = tf.keras.Input((1,), name='discount_price', dtype=tf.float64)
feature_inputs['pay_method'] = tf.keras.Input((1,), name='pay_method', dtype=tf.string)
feature_inputs['comment'] = tf.keras.Input((1,), name='comment', dtype=tf.string)
feature_inputs['comment_num'] = tf.keras.Input((1,), name='comment_num', dtype=tf.int32)
feature_inputs['reward_point'] = tf.keras.Input((1,), name='reward_point', dtype=tf.float64)
feature_inputs['use_reward_point'] = tf.keras.Input((1,), name='use_reward_point', dtype=tf.float64)
feature_inputs['level'] = tf.keras.Input((1,), name='level', dtype=tf.string)
feature_inputs['order_range'] = tf.keras.Input((1,), name='order_range', dtype=tf.string)
feature_inputs['r'] = tf.keras.Input((1,), name='r', dtype=tf.string)
feature_inputs['f'] = tf.keras.Input((1,), name='f', dtype=tf.string)
feature_inputs['m'] = tf.keras.Input((1,), name='m', dtype=tf.string)
feature_inputs['newcustomer'] = tf.keras.Input((1,), name='newcustomer', dtype=tf.string)
feature_inputs['ordercustomer'] = tf.keras.Input((1,), name='ordercustomer', dtype=tf.string)
feature_inputs['datediff_day'] = tf.keras.Input((1,), name='datediff_day', dtype=tf.string)
feature_inputs['customer_type'] = tf.keras.Input((1,), name='customer_type', dtype=tf.string)
feature_inputs['customer_type_finance'] = tf.keras.Input((1,), name='customer_type_finance', dtype=tf.string)
feature_inputs['pro_type'] = tf.keras.Input((1,), name='pro_type', dtype=tf.string)
feature_inputs['product_color'] = tf.keras.Input((1,), name='product_color', dtype=tf.string)
feature_inputs['product_texture'] = tf.keras.Input((1,), name='product_texture', dtype=tf.string)
feature_inputs['lace_size'] = tf.keras.Input((1,), name='lace_size', dtype=tf.string)
feature_inputs['wigtype'] = tf.keras.Input((1,), name='wigtype', dtype=tf.string)
feature_inputs['show_sum'] = tf.keras.Input((1,), name='show_sum', dtype=tf.int32)
feature_inputs['click_sum'] = tf.keras.Input((1,), name='click_sum', dtype=tf.int32)
feature_inputs['add_sum'] = tf.keras.Input((1,), name='add_sum', dtype=tf.int32)
feature_inputs['order_sum'] = tf.keras.Input((1,), name='order_sum', dtype=tf.int32)
feature_inputs['ctr'] = tf.keras.Input((1,), name='ctr', dtype=tf.float64)
feature_inputs['atr'] = tf.keras.Input((1,), name='atr', dtype=tf.float64)
feature_inputs['cvr'] = tf.keras.Input((1,), name='cvr', dtype=tf.float64)
# feature_inputs['acvr'] = tf.keras.Input((1,), name='acvr', dtype=tf.int64)

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
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# model
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
inputs = feature_layer(feature_inputs)
# inputs = tf.keras.layers.DenseFeatures(feature_columns)(feature_inputs)

model_dir = "unice_rank_model_v3_1"
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



else:
  print("load unice_rank_model v3..........")
  model = tf.keras.models.load_model(model_dir)

EPOCHS = 2
# model = build_model(input_shape, output_shape)
loss_object = tf.keras.losses.BinaryCrossentropy(reduction='auto', name='binary cross entropy')
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.BinaryAccuracy()
test_accuracy = tf.keras.metrics.BinaryAccuracy()

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

def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train[0], predictions[0]) + loss_object(y_train[1], predictions[1])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test[0], predictions[0]) + loss_object(y_test[1], predictions[1])
    test_loss(loss)

log_dir="log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = log_dir + '/train'
test_log_dir = log_dir + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(EPOCHS):
    for (x_train, (y1, y2)) in train_ds:
        train_step(model, optimizer, x_train, (y1, y2))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for (x_test, (y1, y2)) in test_ds:
        test_step(model, x_test, (y1, y2))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
# summary
# tensorboard --logdir log
# log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# history = model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=2,
#           verbose=1,
#           callbacks=[tensorboard_callback])


# import matplotlib.pyplot as plt
#
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
# loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(test_ds)
# print("pctr_accuracy", pctr_accuracy)
# print("pcvr_accuracy", pcvr_accuracy)
#
# print(model.predict(test_ds))

# model.save(model_dir)


t3 = time.time()
print("执行总花费时间： {}".format(t3 - t1))
print("数据预处理花费时间： {}".format(t2 - t1))
print("模型训练花费时间： {}".format(t3 - t2))
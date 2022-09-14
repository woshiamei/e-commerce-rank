from typing import List

import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import logging
import random
from joblib import Parallel, delayed

import tensorflow as tf

# from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from utils.connect import hive_connect

import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--date', type=str, default='20220807')
parse.add_argument('--jobs_index', type=int, default=1)
parse.add_argument('--jobs_num', type=int, default=4)
parse.add_argument('--per_vids_num', type=int, default=10)
parse.add_argument('--per_insert_num', type=int, default=100)
parse.add_argument('--model_path_prefix', type=str, default='unice_rank_model_v7_1')
args = parse.parse_args()

# 第二：读取数据之后，对每个模块（分为 user info 、 item info）的数据进行预处理（一定要和训练保持一致）
# 第三：将处理好的数据进行拼接，然后分批次进行预估（如果速度较慢，可尝试启动多个预估程序，可以按照 vid 来划分任务）

# logging.debug("debug logs")

batch_size = 512
w_pctr = 1
w_pcvr = 1
topN_1 = 100
topN = 30
smooth = 0.01
per_vids_num = args.per_vids_num
per_insert_num = args.per_insert_num
model_path_prefix = args.model_path_prefix

jobs_index = args.jobs_index
jobs_num = args.jobs_num
hdb = hive_connect()

date = datetime.datetime.strptime(args.date, '%Y%m%d')
cur_time = datetime.datetime.strptime(args.date, '%Y%m%d').strftime("%Y%m%d")

file_vids_to_uids = "../pred_data/file_vids_to_uids_{}.csv".format(cur_time)
file_pids_feas = "../pred_data/file_pids_feas_{}.csv".format(cur_time)
file_users_feas = "../pred_data/file_users_feas_{}.csv".format(cur_time)
file_vids_pids_sum = "../pred_data/file_vids_pids_sum_{}.csv".format(cur_time)

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


def common_predict(data, num):
	print('common predict, num: {}'.format(num))

	t1 = time.time()
	print("model {}..........".format(model))
	# model = tf.keras.models.load_model(model_dir)
	data_ds = df_to_dataset(data, batch_size=batch_size)

	# loss, pctr_loss, pcvr_loss, pctr_accuracy, pcvr_accuracy = model.evaluate(data_ds)
	# print("pctr_accuracy", pctr_accuracy)
	# print("pcvr_accuracy", pcvr_accuracy)
	res = model.predict(data_ds)
	print("res: {}".format(res))
	# 转 df
	pred_res_df = pd.DataFrame()
	pred_res_df[['vid', 'uid', 'pid']] = data[['fullvisitorid', 'id', 'productsku']]
	pred_res_df[['pctr', 'pcvr']] = pd.DataFrame(np.concatenate([res[0], res[1]], axis=1))
	# 将 多个label 拼在一起
	pred_res_df.columns = ['vid', 'uid', 'pid', 'pctr', 'pcvr']

	# 预估结果集太大， 每个用户暂定 100
	# 算一个加权分
	# pred_res_df['score'] = pred_res_df.apply(lambda x: x.pctr * w_pctr + x.pcvr * w_pcvr, axis=1)

	# 先去掉数据中的nan
	pred_res_df = pred_res_df.dropna()

	# 对数据按A列分组并在组内按B列排序
	# pred_res_df = pred_res_df.groupby(by='vid').apply(lambda x: x.sort_values('score', ascending=False))

	# # ensemble sort
	pred_res_df = pred_res_df.groupby(by='vid').apply(lambda x: x.sort_values('pctr', ascending=False))
	pred_res_df = pred_res_df.reset_index(drop=True)
	pred_res_df['rank_pctr'] = pred_res_df.groupby(by='vid').cumcount() + 1
	pred_res_df['score_pctr'] = pred_res_df.apply(lambda x: 1.0 / (x.rank_pctr + smooth), axis=1)
	# pred_res_df.drop(['rank'], axis=1, inplace=True)
	pred_res_df = pred_res_df.groupby(by='vid').apply(lambda x: x.sort_values('pcvr', ascending=False))
	pred_res_df = pred_res_df.reset_index(drop=True)
	pred_res_df['rank_pcvr'] = pred_res_df.groupby(by='vid').cumcount() + 1
	pred_res_df['score_pcvr'] = pred_res_df.apply(lambda x: 1.0 / (x.rank_pcvr + smooth), axis=1)

	pred_res_df['score'] = pred_res_df.apply(lambda x: x.score_pctr * w_pctr + x.score_pcvr * w_pcvr, axis=1)
	pred_res_df.drop(['rank_pctr', 'score_pctr', 'rank_pcvr', 'score_pcvr'], axis=1, inplace=True)

	pred_res_df = pred_res_df.groupby(by='vid').apply(lambda x: x.sort_values('score', ascending=False))

	# 取分级数据的前N个值
	pred_res_df = pred_res_df.reset_index(drop=True).groupby(by='vid').head(topN_1)

	# print("pred_res_df: ")
	# print(pred_res_df.describe())
	# pred_res_file = '../pred_data/pred_res_{}_{}.csv'.format(cur_time, num)
	# pred_res_df.to_csv(pred_res_file, index=False)

	# 在这里存入数据库
	# 离线方式： 插入
	# 线上方式： 更新

	base_sql_2 = 'insert into gdm.s_bq_user_item_score (vid, uid, pid, pctr, pcvr, `date`) VALUES {}'
	t21 = time.time()
	num = 0
	data_list = []
	for i in pred_res_df.itertuples():
		vid = i[1]
		uid = i[2]
		pid = i[3]
		pctr = str(round(i[4], 4))
		pcvr = str(round(i[5], 4))
		date = cur_time
		data_list.append((vid, uid, pid, pctr, pcvr, date))
		if num % per_insert_num == 0 and num != 0:
			t22 = time.time()
			print('insert--- {} --- {}s'.format(num, t22 - t21))
			sql = base_sql_2.format(','.join(str(item) for item in data_list))
			hdb.insert(sql)
			data_list = []
			t21 = time.time()
		num += 1
	sql = base_sql_2.format(','.join(str(item) for item in data_list))
	hdb.insert(sql)

	base_sql = 'upsert into gdm.s_bq_user_reco_top50 (vid, uid, reco_product) VALUES {} '
	vids = pred_res_df['vid'].unique()
	t23 = time.time()
	sep = ","
	num = 0
	data_list = []
	for vid in vids:
		# tmp = list(pred_res_df[pred_res_df['vid'] == vid]['pid'])
		# 排序策略
		tmp = pred_res_df[pred_res_df['vid'] == vid]
		tmp_res = rank_strategy(vid, tmp)
		res = sep.join(str(v) for v in tmp_res)
		uids = pred_res_df[pred_res_df['vid'] == vid]['uid']
		uid = '-1'
		try:
			uid = list(uids)[0]
		except:
			print("uid = list(uids)[0] is error !")

		data_list.append((vid, uid, res))
		if num % per_insert_num == 0 and num != 0:
			t24 = time.time()
			print('insert--- {} --- {}s'.format(num, t24 - t23))
			sql = base_sql.format(','.join(str(item) for item in data_list))
			hdb.upsert(sql)
			data_list = []
			t23 = time.time()
		num += 1
	sql = base_sql.format(','.join(str(item) for item in data_list))
	hdb.upsert(sql)

	t2 = time.time()
	print("一次预估存储执行总花费时间： {}".format(t2 - t1))


window = 10
min_num = 0
max_num = 5
w_show = 0.7
w_click = 3.0
w_add = 1.8
w_order = 1.2
def rank_strategy(vid, cand_df):
	dict_pid_score = dict(zip(cand_df['pid'], cand_df['score']))
	# 将曝光少的产品进行爬坡
	if len(pids_papo) == papo_num:
		# index = 10
		scores = []
		start_insert = 4
		end_insert = window
		for i in range(papo_num):
			end_insert = min(end_insert, topN - 1)
			start_insert = min(start_insert, end_insert)
			index = random.randint(start_insert, end_insert)
			try:
				scores.append(cand_df.iloc[index, 5])
			except:
				print(" papo pid scores get error! ")

			start_insert = end_insert + 1
			end_insert = start_insert + window

		for i in range(len(scores)):
			dict_pid_score[str(pids_papo[i])] = scores[i]

	# 将用户 + 产品的统计值进行 提权和打压
	if vid in vids_unique:
		tmp = vids_pids_sum_df[vids_pids_sum_df['vid'] == vid]
		# print(tmp)
		tmp_pids_unique = tmp['pid'].unique()
		for k, v in dict_pid_score.items():
			if str(k) in tmp_pids_unique:
				record = list(tmp[tmp.pid == str(k)].iloc[0, :])
				show_sum = max(min_num, min(record[2], max_num))
				click_sum = max(min_num, min(record[3], max_num))
				add_sum = max(min_num, min(record[4], max_num))
				order_sum = max(min_num, min(record[5], max_num))
				w = (w_show ** show_sum) * (w_click ** click_sum) * (w_add ** add_sum) * (w_order ** order_sum)
				dict_pid_score[k] = w * v

	# 排序
	res = list(sorted(dict_pid_score.items(), key=lambda k: k[1], reverse=True))[:topN]
	res = list([str(sim_pair[0]) for sim_pair in res])
	return res


def pre_handle(vids_to_uids_df, uids_feas_df, pids_feas_df):
	# 迁移到 common predict

	# 检查 id 类型是否一致
	# vids_to_uids_df.id = vids_to_uids_df['id'].astype(int)
	user_info = pd.merge(vids_to_uids_df, uids_feas_df, on='id', how='left')

	# user_info[user_info['id'].notna()]

	# train predict 一致性
	# uids_feas_df.id = uids_feas_df['id'].astype(float)
	# uids_feas_df['id'] = uids_feas_df['id'].fillna('-1')
	# pids_feas_df.productsku = pids_feas_df['productsku'].astype(int)
	# pids_feas_df['productsku'] = pids_feas_df['productsku'].fillna('-1')
	# user_info['id'].fillna('-1', inplace=True)
	# pids_feas_df['productsku'].fillna('-1', inplace=True)
	# vids_to_uids_df.fillna('-1', inplace=True)

	def predict_data_handle(data, type='user'):
		if type == 'item':
			# 临时使用 0 对 xtr 进行填充，进行测试
			# for xtr in ['ctr', 'atr', 'cvr']:
			# 	data[xtr] = data[xtr].fillna(0.0)
			# xtr 使用均值进行填充
			mean = data["ctr"].mean()
			data['ctr'].fillna(mean, inplace=True)
			data.loc[:, "ctr"] = data.apply(lambda x: x.ctr * 1e2, axis=1)
			mean = data["atr"].mean()
			data['atr'].fillna(mean, inplace=True)
			data.loc[:, "atr"] = data.apply(lambda x: x.atr * 1e2, axis=1)
			# data["atr"] = data.apply(lambda x: x.atr * 1e2 if x.atr is not None else mean * 1e2, axis=1)
			mean = data["cvr"].mean()
			data['cvr'].fillna(mean, inplace=True)
			data.loc[:, "cvr"] = data.apply(lambda x: x.cvr * 1e2, axis=1)
		# data["cvr"] = data.apply(lambda x: x.cvr * 1e2 if x.cvr is not None else mean * 1e2, axis=1)

		if type == 'user':
			# data['order_latest_time'] = data['order_latest_time'].apply(
			#     lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '2000-01-01')
			# data['order_first_time'] = data['order_first_time'].apply(
			#     lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '2000-01-01')
			# data['order_latest_time'] = data['order_latest_time'].astype(str)
			# data['order_first_time'] = data['order_first_time'].astype(str)
			# data.order_latest_time = data.apply(lambda x : (datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_latest_time, '%Y-%m-%d')).days, axis=1)
			# data.order_first_time = data.apply(lambda x : (datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_first_time, '%Y-%m-%d')).days, axis=1)

			dict1 = {'order_first_time': '2000-01-01 00:00:00', 'order_latest_time': '2000-01-01 00:00:00'}
			data.fillna(dict1, inplace=True)
			data.loc[:, "order_latest_time"] = data.apply(lambda x: (
					datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_latest_time,
					                                                                            '%Y-%m-%d %H:%M:%S')).days,
			                                              axis=1)
			data.loc[:, "order_first_time"] = data.apply(lambda x: (
					datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.datetime.strptime(x.order_first_time,
					                                                                            '%Y-%m-%d %H:%M:%S')).days,
			                                             axis=1)

		data.loc[:, :] = data.apply(lambda x: x.fillna(-1) if x.dtype.kind in 'biufc' else x.fillna('-'))
		if type == 'user':
			for fea in ['orders_count', 'orders_month', 'orders_3month', 'orders_6month', 'orders_year', 'discount_num',
			            'comment_num']:
				data.loc[:, fea] = data.loc[:, fea].astype(int)
			for fea in ['fullvisitorid', 'id', 'gender']:
				if data[fea].dtype.kind in 'f':
					data.loc[:, fea] = data.loc[:, fea].astype(int)
				data.loc[:, fea] = data.loc[:, fea].astype(str)
		else:
			for fea in ['show_sum', 'click_sum', 'add_sum', 'order_sum']:
				data.loc[:, fea] = data.loc[:, fea].astype(int)
			for fea in ['productsku', 'lace_size', 'pro_type', 'product_color', 'product_texture', 'wigtype']:
				if data[fea].dtype.kind in 'f':
					data.loc[:, fea] = data.loc[:, fea].astype(int)
				data.loc[:, fea] = data.loc[:, fea].astype(str)
		print('{} data handle num is {}'.format(type, len(data)))

		return data

	user_info = predict_data_handle(user_info, 'user')

	pids_feas_df = predict_data_handle(pids_feas_df, 'item')
	# vid -> uid
	# for fea in ['fullvisitorid', 'id']:
	# 	vids_to_uids_df[fea] = vids_to_uids_df[fea].astype(str)

	return user_info, pids_feas_df


# 全局的爬坡候选集
def papo(pid_no_xtr_df):
	# print("pid_no_xtr_df: ")
	# print(pid_no_xtr_df.describe())
	global pids_papo
	global papo_num
	pids_papo = []
	papo_num = 2
	# 第一版先简单点，每个进程随机挑选出两个
	random.seed(jobs_index)
	pids = list(pid_no_xtr_df.productsku)
	# candidate_index = []
	if len(pids) <= 2:
		print('爬坡候选集个数小于等于 2， num： {}'.format(len(pids)))
		pids_papo = pids
	while len(pids_papo) < papo_num:
		pid = pids[random.randint(0, len(pids) - 1)]
		if pid in pids_papo:
			continue
		pids_papo.append(pid)


def sample_struct(num):
	print("sample_struct 线程数： {}".format(num + 1))
	# 分批次进行处理
	# 该进程的处理的行数
	global vids_to_uids_df
	print("vids_to_uids_df shape： ")
	print(vids_to_uids_df.shape)
	jobs_index = num + 1
	row_start = int(vids_to_uids_df.shape[0] * ((jobs_index - 1) / float(jobs_num)))
	row_end = int(vids_to_uids_df.shape[0] * (jobs_index / float(jobs_num)))
	vids_to_uids_df = vids_to_uids_df.iloc[row_start: row_end, :]

	print("本次处理的行数： {} -- {}".format(row_start, row_end))
	print("本次需要处理的 vids num: {}".format(vids_to_uids_df.shape[0]))
	print("uids num: {}".format(uids_feas_df.shape[0]))

	# 进行数据预处理
	user_info, pids_feas_df = pre_handle(vids_to_uids_df, uids_feas_df, pids_xtr_df)
	vids_set = user_info['fullvisitorid'].unique()

	print('pids num: {}'.format(len(pids_set)))
	# sep = ','
	tmp_data = []
	i = 0
	j = 0
	for vid in vids_set:
		if vid == "-1":
			continue
		i += 1
		for pid in pids_set:
			# print(user_info[user_info.vid == vid])
			user = user_info[user_info.fullvisitorid == vid]
			# print(pids_feas_df[pids_feas_df.pid == pid])
			item = pids_feas_df[pids_feas_df.productsku == str(pid)]
			# print(np.hstack((user, item)))
			# if user.shape[0] != item.shape[0]:
			if len(user) != len(item):
				j += 1
				# print("user shape {} is not match item shape {}!".format(len(user), len(item)))
				continue
			tmp = np.hstack((user, item))
			tmp_data.append(tmp[0])
		# print('size not match num: {}'.format(j))

		# mod = int(vid) % 2
		# files_map[mod].write(sep.join(str(i) for i in tmp[0]) + '\n')
		if i % per_vids_num == 0 and i > 0:
			pred_data = pd.DataFrame(tmp_data)
			pred_data.columns = ['fullvisitorid', 'id',
			                     'gender', 'age_range', 'country', 'region', 'city', 'register_channel',
			                     'register_channel_type',
			                     'register_type_one', 'register_type_two', 'orders_count', 'totalspent', 'maxtotal',
			                     'atv_range', 'order_first_time', 'order_latest_time',
			                     'orders_month', 'orders_3month', 'orders_6month',
			                     'orders_year', 'discount_num', 'discount_price', 'pay_method', 'comment',
			                     'comment_num',
			                     'reward_point', 'use_reward_point', 'level', 'order_range', 'r', 'f', 'm',
			                     'newcustomer', 'ordercustomer',
			                     'datediff_day', 'customer_type', 'customer_type_finance',
			                     'productsku', 'pro_type', 'product_color', 'product_texture', 'lace_size', 'wigtype',
			                     'show_sum', 'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr']

			common_predict(pred_data, i // per_vids_num)
			# init
			tmp_data = []

	if len(tmp_data) > 100:
		pred_data = pd.DataFrame(tmp_data)
		pred_data.columns = ['fullvisitorid', 'id',
		                     'gender', 'age_range', 'country', 'region', 'city', 'register_channel',
		                     'register_channel_type',
		                     'register_type_one', 'register_type_two', 'orders_count', 'totalspent', 'maxtotal',
		                     'atv_range', 'order_first_time', 'order_latest_time',
		                     'orders_month', 'orders_3month', 'orders_6month',
		                     'orders_year', 'discount_num', 'discount_price', 'pay_method', 'comment',
		                     'comment_num',
		                     'reward_point', 'use_reward_point', 'level', 'order_range', 'r', 'f', 'm',
		                     'newcustomer', 'ordercustomer',
		                     'datediff_day', 'customer_type', 'customer_type_finance',
		                     'productsku', 'pro_type', 'product_color', 'product_texture', 'lace_size', 'wigtype',
		                     'show_sum', 'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr']

		common_predict(pred_data, i // per_vids_num + 1)

	print('size not match num: {}'.format(j))
	t3 = time.time()
	print("jobs_index: {} ， 整个预估过程耗时： {}".format(jobs_index, t3 - t1))

def predict_run():
	t1 = time.time()

	global vids_to_uids_df
	global pids_feas_df
	global uids_feas_df
	# 读入数据
	vids_to_uids_df = pd.read_csv(file_vids_to_uids)
	pids_feas_df = pd.read_csv(file_pids_feas)
	uids_feas_df = pd.read_csv(file_users_feas)
	uids_feas_df.drop(['email'], axis=1, inplace=True)
	print("all vids num: {}".format(vids_to_uids_df.shape[0]))

	# 读取 vids_pids_sum , global
	global vids_pids_sum_df
	global vids_unique
	vids_pids_sum_df = pd.read_csv(file_vids_pids_sum, dtype={'fullvisitorid': object, 'productsku': object})
	vids_pids_sum_df.columns = ['vid', 'pid', 'show_sum', 'click_sum', 'add_sum', 'order_sum']
	print("vids pids sum num: {}".format(vids_pids_sum_df.shape[0]))

	vids_unique = vids_pids_sum_df['vid'].unique()
	print("distinct vids num : {} ".format(len(vids_unique)))

	# 对 show num < 1000 的产品进行策略调整，不直接进行预估
	# 转换成对 xtr 为空的 pids 不进行预估
	pids_set_normal = pids_feas_df['productsku'].unique()
	print("在售产品的个数：{}".format(len(pids_set_normal)))
	global pids_xtr_df
	pids_xtr_df = pids_feas_df.dropna(subset=['ctr', 'atr', 'cvr'])
	global pids_set
	pids_set = pids_xtr_df['productsku'].unique()
	print("正常在售且曝光大于 1000 的产品的个数： {}".format(len(pids_set)))

	# 处理曝光少的产品： 选取部分产品作为候选，后面爬坡使用
	pid_no_xtr = list(set(pids_set_normal) - set(pids_set))
	print("爬坡产品个数： {}".format(len(pid_no_xtr)))
	pid_no_xtr_df = pids_feas_df[pids_feas_df['productsku'].isin(pid_no_xtr)]
	papo(pid_no_xtr_df)

	t2 = time.time()
	print("第一步读数据耗时： {}".format(t2 - t1))

	# 在这里进行多线程
	repetitions = jobs_num
	Parallel(n_jobs=repetitions)(delayed(sample_struct)(i) for i in range(repetitions))

def checkout_model(version="unice_rank_model_v3"):
	days_num = 10
	for i in range(days_num):
		model_time = (datetime.datetime.strptime(cur_time, '%Y%m%d') - datetime.timedelta(days=i)).strftime("%Y%m%d")
		model_dir = "../model/{}_{}".format(version, model_time)
		if os.path.exists(model_dir):
			break
		else:
			print("model dir: {} not exist!".format(model_dir))
			if i == days_num - 1:
				print("exit !!!!!!!")
				exit()
			continue
	return model_dir


if __name__ == '__main__':
	print("common predict starting **************")

	t1 = time.time()
	print("jobs_index: {}, jobs_num: {}".format(jobs_index, jobs_num))
	print("per_vids_num: {}, per_insert_num: {}".format(per_vids_num, per_insert_num))

	# 清空表
	# print("清空表：gdm.s_bq_user_item_score， delete gdm.s_bq_user_reco_top50 ")
	# hdb.delete('delete gdm.s_bq_user_item_score')
	# hdb.delete('delete gdm.s_bq_user_reco_top50')

	# model 只 load 一次
	model_dir = checkout_model(model_path_prefix)
	print("load unice_rank_model {}..........".format(model_dir))
	model = tf.keras.models.load_model(model_dir)

	predict_run()

	tf.keras.backend.clear_session()

	t2 = time.time()
	print("该预估总耗时： {}".format(t2 - t1))

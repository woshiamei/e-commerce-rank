# 将结果写到接口里面
# from urllib import request,parse

import requests
import json
import time
import datetime
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from utils.connect import hive_connect

import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='get')
# parser.add_argument('--bb', type=int, default=32)
args = parser.parse_args()

delta = 90
item_delta = 40
start_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
cur_time = (datetime.datetime.now()).strftime("%Y-%m-%d").replace("-", "")
item_start_time = (datetime.datetime.now() - datetime.timedelta(days=item_delta)).strftime("%Y-%m-%d").replace("-", "")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)

# 用户维度的
# global hdb
hdb = hive_connect()
host = '192.168.1.150'

# 将默认候选集赋值给 unice_0000
# 并更新到 hive 表
topN = 30
def post_default_topN_data():
    global hdb
    item_info_res =  hdb.get_all("""
        SELECT productsku, 
        sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as show_sum, 
        sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) as click_sum, 
        sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) as add_sum,
        sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) as order_sum,
        sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as ctr,
        sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as atr,
        sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as cvr
        from ods.s_bq_custAct
        where `date` > '{}'
        and productsku regexp '^[0-9]+$'
        GROUP BY productsku
        having sum(cast(p_view as BIGINT)) > 1000
        order by  sum(cast(p_view as BIGINT)) DESC
    """.format(item_start_time))

    print("len item_info_res : {}".format(len(item_info_res)))
    item_info = pd.DataFrame(list(item_info_res))
    item_info.columns = ["pid", 'show_sum', 'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr']

    item_info = item_info[item_info.cvr < 0.2]
    item_info = item_info[item_info.atr < 0.2]
    item_info = item_info[item_info.ctr < 0.2]
    item_info = item_info[item_info.show_sum > 3000]

    w_ctr, w_atr, w_cvr = 0.3, 0.3, 0.4
    item_info['score'] = item_info.apply(lambda x: w_ctr * x.ctr + w_atr * x.atr + w_cvr * x.cvr, axis=1)

    item_info.sort_values("score", ascending=False, inplace=True)

    # print(item_info.describe())
    print(item_info.head())

    res_xtr = list(item_info.pid)[:200]
    print("res_xtr: {}".format(res_xtr))

    # normal pids
    normal_pids_res = hdb.get_all(
        """
            SELECT
                `e`.`entity_id` 
            FROM
                `s_sc_catalog_product_flat_1_unice` AS `e`
                LEFT JOIN `s_sc_cataloginventory_stock_status_unice` AS `i` ON e.entity_id = i.product_id 
            WHERE
                ( e.STATUS = 1 ) 
                AND ( e.price > 1 ) 
                AND ( e.entity_id IN ( SELECT entity_id FROM s_sc_catalog_product_entity_int_unice WHERE `value` = 824 GROUP BY entity_id ) ) 
                AND ( i.qty > 0 ) 
                AND ( i.stock_status = 1 ) 
            GROUP BY
                `e`.`entity_id` 
            ORDER BY
                `e`.`entity_id` ASC
        """)

    normal_pids_file = "../normal_pids.csv"

    # 判断 normal 个数
    if len(normal_pids_res) < 300:
        print("len normal pids is too short, cur time: {}, need checking!!!!!!!!!!!!!".format(cur_time))
        # exit()
    else:
        normal_pids_df = pd.DataFrame(list(normal_pids_res))
        normal_pids_df.columns = ['pid']
        # data_to_csv(raw_item_info, item_info_file)
        normal_pids_df.to_csv(normal_pids_file, index=False)

    normal_product_id = pd.read_csv(normal_pids_file)


    normal_product_id.columns = ['pid']
    normal_pid_list = list(map(str, list(normal_product_id.pid)))
    print('normal pids num is : {}'.format(len(normal_product_id)))

    xtr_topN = [i for i in res_xtr if i in normal_pid_list][:topN]
    print("len xtr_topN: {}".format(len(xtr_topN)))
    print("xtr_topN: {}".format(xtr_topN))

    # 写入 hive
    base_sql = 'upsert into gdm.s_bq_user_reco_top50 (vid, uid, reco_product) VALUES {} '
    cid = '0000'
    uid = '-1'
    sep = ','
    res = sep.join(str(v) for v in xtr_topN)
    data_list = []
    data_list.append((cid, uid, res))
    sql = base_sql.format(','.join(str(item) for item in data_list))
    # global hdb
    status_cnt = hdb.upsert(sql)
    print("status_cnt: ", status_cnt)
    if status_cnt == 0:
        time.sleep(10)
        hdb.close()
        del hdb
        time.sleep(30)
        hdb = hive_connect()

    # 写入接口
    datas = [{"store": "unice", "cid": cid, "vid": cid, "top_list": res}]
    p_data = json.dumps(datas)
    url = 'http://{}:8000/user_reco/set_user_top/'.format(host)
    a = 0
    i = 0
    while a != 200 and i < 10:
        i += 1
        try:
            response = requests.post(url, data={"data": p_data})
            a = response.status_code
        except Exception as e:
            time.sleep(1)
            print(e)
    print('deafult--{}--{}'.format(len(datas), a))

def post_all_user_topN_data():
    print("***************************************** old user *********************")

    res_list =  hdb.get_all("""
        select vid, uid, reco_product
        from gdm.s_bq_user_reco_top50
        where uid != '-1'
        -- limit 100
    """)

    print("len res_list : {}".format(len(res_list)))
    vids_res_df = pd.DataFrame(list(res_list))
    vids_res_df.columns = ['vid', 'uid', 'res']

    vid_cid_list =  hdb.get_all("""
        SELECT fullvisitorid, clientid
        -- select max(`date`)
        from  gdm.s_bq_vid_to_cid_map
        where `date` > '{}'
        GROUP BY fullvisitorid, clientid
    """.format(start_time))

    print("len vid to cid : {}".format(len(vid_cid_list)))
    vid_cid_df = pd.DataFrame(list(vid_cid_list))
    vid_cid_df.columns = ['vid', 'cid']

    # 拼接
    vid_cid_res = pd.merge(vids_res_df, vid_cid_df, on='vid', how='inner')
    print(vid_cid_res.shape)

    # datas = [{"store": "unice", "vid": item[0], "uid": item[1], "top_list": item[2]} for item in res_list]
    datas = [{"store": "unice", "cid": item.cid, "vid": item.vid, "top_list": item.res} for item in vid_cid_res.itertuples()]
    cut_num = 100

    if len(datas) > cut_num:
        batch_step = len(datas) // cut_num
        n = 1
        for index in range(0, len(datas), batch_step):
            item_list = datas[index:index + batch_step]
            p_data = json.dumps(item_list)
            url = 'http://{}:8000/user_reco/set_user_top/'.format(host)
            a = 0
            i = 0
            while a != 200 and i < 10:
                i += 1
                try:
                    response = requests.post(url, data={"data": p_data})
                    a = response.status_code
                except Exception as e:
                    time.sleep(1)
                    print(e)
            print('{}--{}--{}'.format(len(datas), batch_step * n, a))
            n += 1
    else:
        print("data num {}  < {}".format(len(datas), cut_num))

    print("***************************************** new user *********************")

    res_new_list = hdb.get_all("""
        select vid, uid, reco_product
        from gdm.s_bq_user_reco_top50
        where uid = '-1'
        -- limit 100
    """)

    print("len res_new_list : {}".format(len(res_new_list)))
    print("len vid to cid : {}".format(len(vid_cid_list)))
    vids_res_df = pd.DataFrame(list(res_new_list))
    vids_res_df.columns = ['vid', 'uid', 'res']
    # 拼接
    vid_cid_res = pd.merge(vids_res_df, vid_cid_df, on='vid', how='inner')
    print(vid_cid_res.shape)

    datas = [{"store": "unice", "cid": item.cid, "vid": item.vid, "top_list": item.res} for item in
             vid_cid_res.itertuples()]
    cut_num = 1000

    if len(datas) > cut_num:
        batch_step = len(datas) // cut_num
        n = 1
        for index in range(0, len(datas), batch_step):
            item_list = datas[index:index + batch_step]
            p_data = json.dumps(item_list)
            url = 'http://{}:8000/user_reco/set_user_top/'.format(host)
            a = 0
            i = 0
            while a != 200 and i < 10:
                i += 1
                try:
                    response = requests.post(url, data={"data": p_data})
                    a = response.status_code
                except Exception as e:
                    time.sleep(1)
                    print(e)
            print('{}--{}--{}'.format(len(datas), batch_step * n, a))
            n += 1
    else:
        print("data num {}  < {}".format(len(datas), cut_num))

def post_old_user_topN_data():
    res_list =  hdb.get_all("""
        select vid, uid, reco_product
        from gdm.s_bq_user_reco_top50
        where uid != '-1'
        -- limit 100
    """)

    datas = [{"store": "unice", "vid": item[0], "uid": item[1], "top_list": item[2]} for item in res_list]
    cut_num = 100

    if len(datas) > cut_num:
        batch_step = len(datas) // cut_num
        n = 1
        for index in range(0, len(datas), batch_step):
            item_list = datas[index:index + batch_step]
            p_data = json.dumps(item_list)
            url = 'http://{}:8000/user_reco/set_user_top/'.format(host)
            a = 0
            i = 0
            while a != 200 and i < 10:
                i += 1
                try:
                    response = requests.post(url, data={"data": p_data})
                    a = response.status_code
                except Exception as e:
                    time.sleep(1)
                    print(e)
            print('{}--{}--{}'.format(len(datas), batch_step * n, a))
            n += 1
    else:
        print("data num {}  < {}".format(len(datas), cut_num))

def post_new_user_topN_data():
    res_list =  hdb.get_all("""
        select vid, uid, reco_product
        from gdm.s_bq_user_reco_top50
        where uid = '-1'
        -- limit 100
    """)

    datas = [{"store": "unice", "vid": item[0], "uid": item[1], "top_list": item[2]} for item in res_list]
    cut_num = 1000

    if len(datas) > cut_num:
        batch_step = len(datas) // cut_num
        n = 1
        for index in range(0, len(datas), batch_step):
            item_list = datas[index:index + batch_step]
            p_data = json.dumps(item_list)
            url = 'http://{}:8000/user_reco/set_user_top/'.format(host)
            a = 0
            i = 0
            while a != 200 and i < 10:
                i += 1
                try:
                    response = requests.post(url, data={"data": p_data})
                    a = response.status_code
                except Exception as e:
                    time.sleep(1)
                    print(e)
            print('{}--{}--{}'.format(len(datas), batch_step * n, a))
            n += 1
    else:
        print("data num {}  < {}".format(len(datas), cut_num))


def set_vid_data():

    # datas = {}
    # for i in pro_list:
    #     datas[i[0]] = i[1].split(',')
    datas = {'1': ['2', '3', '4']}
    vid_data = json.dumps({"store": "unice", "vid": "123", "value": "1,2,4,6"})
    if vid_data:
        # host = '127.0.0.1'
        host = '192.168.1.150'
        url = 'http://{}:8000/cache_set/'.format(host)
        a = 0
        while a != 200:
            try:
                response = requests.post(url, data={"data": vid_data})
                a = response.status_code
            except Exception as e:
                time.sleep(1)
                print(e)
        print('{}----{}(success)'.format(len(datas), a))
        print('{}----{}(success)'.format(response.content, a))
        p_data = []
    else:
        print("no data to upload")

def set_vid_data_bak():
    # datas = {}
    # for i in pro_list:
    #     datas[i[0]] = i[1].split(',')
    datas = {'1': ['2', '3', '4']}
    vid_data = json.dumps({"store": "unice", "vid": "123", "value": "1,2,4,6"})
    if vid_data:
        # host = '127.0.0.1'
        host = '192.168.1.150'
        url = 'http://{}:8000/cache_set/'.format(host)
        a = 0
        while a != 200:
            try:
                response = requests.post(url, data={"data": vid_data})
                a = response.status_code
            except Exception as e:
                time.sleep(1)
                print(e)
        print('{}----{}(success)'.format(len(datas), a))
        print('{}----{}(success)'.format(response.content, a))
        p_data = []
    else:
        print("no data to upload")


def get_vid_data():
    # datas = {}
    # for i in pro_list:
    #     datas[i[0]] = i[1].split(',')
    # datas = {'1': ['2', '3', '4']}
    # vid_data = json.dumps({"store": "unice", "vid": "123", "value": "1,2,4,6"})
    # if vid_data:
        # host = '127.0.0.1'
    host = '192.168.1.150'
    url = 'http://{}:8000/cache_get/'.format(host)
    a = 0
    while a != 200:
        try:
            # response = requests.post(url, data={"data": vid_data})
            response = requests.get(url, params={"store": 'unice', "vid": "123"})
            a = response.status_code
        except Exception as e:
            time.sleep(1)
            print(e)
    # print('{}----{}(success)'.format(len(datas), a))
    print('{}----{}(success)'.format(response.content, a))
    # p_data = []
    # else:
    #     print("no data to upload")

# set_vid_data()
# get_vid_data()


#host= '127.0.0.1'
# host= '172.30.145.102'

def set_data(datas):
    # datas = {}
    # for i in pro_list:
    #     datas[i[0]] = i[1].split(',')
    # datas = {'1': ['2', '3', '4']}
    print("datas: ", datas)
    cos_data = json.dumps({"store": "unice", "product_sim": datas})
    if datas:
        url = 'http://{}:8000/set_data/'.format(host)
        a = 0
        i = 0
        while a != 200 and i < 10:
            i += 1
            try:
                response = requests.post(url, data={"data": cos_data})
                a = response.status_code
            except Exception as e:
                time.sleep(1)
                print(e)
        print('{}----{}(success)'.format(len(datas), a))
        p_data = []
    else:
        print("no data to upload")


def get_data():
    url = 'http://{}:8000/get_product_sim/'.format(host)

    a = 0
    i = 0
    while a != 200 and i < 10:
        i += 1
        try:
            response = requests.get(url, params={"store": 'unice'})
            a = response.status_code
        except Exception as e:
            time.sleep(1)
            print(e)
    print('{}----{}(success)'.format(len(response.json()), a))
    datas = response.json()
    print(datas)
    print(type(datas))

def post_data():
    datas = {}
    cur_time = (datetime.datetime.now()).strftime("%Y-%m-%d").replace("-", "")
    #cur_time = '20220726'
    file = "../res/final_res_{}.csv".format(cur_time)
    if os.path.exists(file):
        with open(file, "r") as f:
            for line in f.readlines():
                r = line.strip().split(" : ")
                if len(r) == 2:
                    datas[r[0]] = list(r[1].strip().split(','))
                else:
                    print("error res: {}".format(line))
    # print(datas)
        if len(datas) > 350:
            set_data(datas)
        else:
            print("len(datas) is short: {}".format(len(datas)))
            exit()
    else:
        print("file not exit: {}".format(file))

if __name__ == '__main__':
    t1 = time.time()
    print("************************ start post data **********************")
    print('start at : {}'.format(time.ctime()))
    # print(args.type)
    # if args.type == 'post':
    #     post_data()
    # else:
    #     get_data()

    post_default_topN_data()
    post_all_user_topN_data()

    # post_old_user_topN_data()
    # post_new_user_topN_data()
    hdb.close()

    t3 = time.time()
    # print("预处理数据执行总花费时间： {}".format(t3 - t2))
    print("执行总花费时间： {}".format(t3 - t1))

import gc

import pandas as pd
import time
import datetime
import numpy as np

import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')
import sys

sys.path.append("/root/reco/i2i/rank")
from utils.connect import hive_connect

import argparse
"""
多目标建模
"""
parse = argparse.ArgumentParser()
parse.add_argument('--date', type=str, default='20220815')
parse.add_argument('--overwirte', type=bool, default=True)
args = parse.parse_args()

# 筛选出需要 predict 的用户
# 用户： vid 为主
# 产品： pid
# 范围: 近 30 天用户
# 更新频率： 天级别
# 离线方式： 插入
# 线上方式： 更新
# 存储：离线-> 数据库， 线上最好是redis ，也可以是数据库
# 策略： 退场， 数据： vid + pid 统计值

# 2022/08/08
# 预估方案
# 存储整个全量样本不符合实际，需要在构造样本时进行预估，最后只存储vid + pid + score
# 主要是分为两部分：
# 第一：通过 sql 将数据拉下来，进行存储
# 第二：读取数据之后，对每个模块（分为 user info 、 item info）的数据进行预处理（一定要和训练保持一致）
# 第三：将处理好的数据进行拼接，然后分批次进行预估（如果速度较慢，可尝试启动多个预估程序，可以按照 vid 来划分任务）


days = 7
days_items = 60
days_items_match = 100
delta = 0

# if args.date
date = datetime.datetime.strptime(args.date, '%Y%m%d')
cur_time = datetime.datetime.strptime(args.date, '%Y%m%d').strftime("%Y%m%d")
# date = datetime.datetime.now()
cur_date = (date - datetime.timedelta(days=delta)).strftime("%Y%m%d")
start_date = (date - datetime.timedelta(days=days)).strftime("%Y%m%d")
cur_dt = (date - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
start_dt = (date - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
start_date_item = (date - datetime.timedelta(days=days_items)).strftime("%Y%m%d")
start_dt_item = (date - datetime.timedelta(days=days_items)).strftime("%Y-%m-%d")

start_date_item_match = (date - datetime.timedelta(days=days_items_match)).strftime("%Y%m%d")
start_dt_item_match = (date - datetime.timedelta(days=days_items_match)).strftime("%Y-%m-%d")

file_vids_to_uids = "../pred_data/file_vids_to_uids_{}.csv".format(cur_time)
file_pids_feas = "../pred_data/file_pids_feas_{}.csv".format(cur_time)
file_users_feas = "../pred_data/file_users_feas_{}.csv".format(cur_time)
file_vids_pids_sum = "../pred_data/file_vids_pids_sum_{}.csv".format(cur_time)
file_normal_pids = '../pred_data/file_normal_pids.csv'

def raw_data_by_sql():
    print('start sql: ')

    t1 = time.time()

    print("cur_date: {}".format(cur_date))
    print("cur_dt: {}".format(cur_dt))
    print("start_date: {}".format(start_date))
    print("start_dt: {}".format(start_dt))

    # all vids
    vids_to_uids = hdb.get_all(
    """
        select distinct fullvisitorid, userid
        from s_bq_session
        where `date` between '{}' and '{}'
        and platform = '自营站' 
        and brand = 'UNice'
    """.format(start_dt, cur_dt))

        # '''
        # select distinct fullvisitorid, userid
        # from s_bq_session_zong
        # where `date` between '{}' and '{}'
        # and shop_mark = 'unicemall'
        # and platform = '自营站'
        # '''

    # user_info
    uids_feas = hdb.get_all(
    """
        select distinct id, email,
            gender, age_range, country, region, city, register_channel, register_channel_type,
            register_type_one, register_type_two, orders_count, totalspent, maxtotal,
            atv_range, cast(order_first_time as string) as order_first_time,  cast(order_latest_time as string) as order_latest_time,
             orders_month, orders_3month, orders_6month,
            orders_year, discount_num, discount_price, pay_method, `comment`, comment_num,
            reward_point, use_reward_point, level, order_range, r, f, m, newcustomer, ordercustomer,
            datediff_day, customer_type, customer_type_finance
        from dw.dws_customer_entity
        where platfrom = "自营站"
        and shop_mark = 'unicemall'
        and brand = 'UNice'
    """)

    # pid info
    pids_feas = hdb.get_all(
    """
        SELECT t7.product_id, pro_type, product_color, product_texture, lace_size, wigtype
            , show_sum, click_sum, add_sum, order_sum, ctr, atr, cvr
        from (
            select product_id, pro_type, product_color, product_texture, lace_size, wigtype
            from (
                select product_id, pro_type, product_color, product_texture, lace_size, wigtype, 
                row_number() over(partition by product_id order by date_time desc) as rank 
                from jdm.s_bq_produce_summary_qu
                where date_time > '{}'
                and shop_mark = 'unicemall'
                and platfrom = "自营站"
            ) t8
            where rank = 1
        ) t7 LEFT JOIN (
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
            having sum(cast(p_view as BIGINT)) > 3000
        )t9 on t7.product_id = t9.productsku
    """.format(start_dt_item, start_date_item))

    # vid - pid cnt
    vids_pids_sum = hdb.get_all(
        """
            with a as (
                select fullvisitorid, t1.productsku, t1.show_sum, t1.click_sum, t1.add_sum, t1.order_sum
                from (
                    select fullvisitorid, productsku, 
                            sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as show_sum, 
                            sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) as click_sum, 
                            sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) as add_sum,
                            sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) as order_sum
                    from ods.s_bq_custAct
                    where `date` > '{}'
                    and productsku regexp '^[0-9]+$'
                    group by fullvisitorid, productsku
                )t1 join (
                    SELECT cast(e.entity_id as string) as pid
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
                         e.entity_id 
                )t2 on t1.productsku = t2.pid
            )
            
            select t3.fullvisitorid, a.productsku, a.show_sum, a.click_sum, a.add_sum, a.order_sum
            from (
                select distinct fullvisitorid
                    from ods.s_bq_custAct
                    where `date` > '{}'
                    and productsku regexp '^[0-9]+$'
                )t3 left join a
                on t3.fullvisitorid = a.fullvisitorid
                -- limit 10
                ;
        """.format(start_date_item_match, start_date))
    # vids_pids_sum = hdb.get_all(
    #     """
    #     select fullvisitorid, t1.productsku, t1.show_sum, t1.click_sum, t1.add_sum, t1.order_sum
    #         from (
    #             select fullvisitorid, productsku,
    #                     sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as show_sum,
    #                     sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) as click_sum,
    #                     sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) as add_sum,
    #                     sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) as order_sum
    #             from ods.s_bq_custAct
    #             where `date` > '{}'
    #             and productsku regexp '^[0-9]+$'
    #             group by fullvisitorid, productsku
    #             )t1 inner join (
    #                 SELECT productsku
    #                     -- sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as show_sum,
    #                     -- sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) as click_sum,
    #                     -- sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) as add_sum,
    #                     -- sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) as order_sum,
    #                     -- sum(CASE WHEN p_click = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as ctr,
    #                     -- sum(CASE WHEN is_addtocart = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as atr,
    #                     -- sum(CASE WHEN is_order = '0' THEN 0 ELSE 1 END) /  sum(CASE WHEN p_view = '0' THEN 0 ELSE 1 END) as cvr
    #                 from ods.s_bq_custAct
    #                 where `date` > '{}'
    #                 and productsku regexp '^[0-9]+$'
    #                 GROUP BY productsku
    #                 having sum(cast(p_view as BIGINT)) > 1000
    #             ) t2 on t1.productsku = t2.productsku
    #     """.format(start_date_item_match, start_date_item_match))

    print("vids_to_uids: {}".format(len(vids_to_uids)))
    print("uids_feas: {}".format(len(uids_feas)))
    print("pids_feas: {}".format(len(pids_feas)))
    print("vids_pids_sum: {}".format(len(vids_pids_sum)))

    t2 = time.time()
    print("sql 查询耗时： {}".format(t2 - t1))
    # print("count(cnt >= 3): {}".format(user_item_cnt))
    # for i in range(10):
    #     print(user_item_action[i])
    vids_to_uids_df = pd.DataFrame(list(vids_to_uids))
    vids_to_uids_df.columns = ['fullvisitorid', 'id']

    uids_feas_df = pd.DataFrame(list(uids_feas))
    uids_feas_df.columns = [
        'id', 'email',
        'gender', 'age_range', 'country', 'region', 'city', 'register_channel', 'register_channel_type',
        'register_type_one', 'register_type_two', 'orders_count', 'totalspent', 'maxtotal',
        'atv_range', 'order_first_time', 'order_latest_time',
        'orders_month', 'orders_3month', 'orders_6month',
        'orders_year', 'discount_num', 'discount_price', 'pay_method', 'comment', 'comment_num',
        'reward_point', 'use_reward_point', 'level', 'order_range', 'r', 'f', 'm', 'newcustomer', 'ordercustomer',
        'datediff_day', 'customer_type', 'customer_type_finance'
    ]

    pids_feas_df = pd.DataFrame(list(pids_feas))
    pids_feas_df.columns = [
        'productsku', 'pro_type', 'product_color', 'product_texture', 'lace_size', 'wigtype',
        'show_sum', 'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr'
    ]

    vids_pids_sum_df = pd.DataFrame(list(vids_pids_sum))
    vids_pids_sum_df.columns = [
        # t1.click_sum, t1.add_sum, t1.order_sum
        'fullvisitorid', 'productsku', 'show_sum', 'click_sum', 'add_sum', 'order_sum'
    ]

    t2 = time.time()
    print("查询数据执行总花费时间： {}".format(t2 - t1))

    # 数据预处理
    # 正常产品 过滤
    # normal_product_id_file = "/root/reco/i2i/data/product_id_set.xlsx"
    # normal_product_id = pd.read_excel(normal_product_id_file)
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


    if len(normal_pids_res) < 350:
        print("len normal pids is too short, cur time: {}, need checking!!!!!!!!!!!!!".format(cur_time))
        # exit()
    else:
        normal_pids_df = pd.DataFrame(list(normal_pids_res))
        normal_pids_df.columns = ['pid']
        # data_to_csv(raw_item_info, item_info_file)
        normal_pids_df.to_csv(file_normal_pids, index=False)

    normal_product_id = pd.read_csv(file_normal_pids)
    # normal_product_id = pd.DataFrame(list(normal_pids_res))

    normal_product_id.columns = ['entity_id']
    normal_pid_list = list(map(str, list(normal_product_id.entity_id)))
    # normal_pid_set = set(map(str, set(normal_product_id.entity_id)))
    print("normal product num： {}".format(len(normal_pid_list)))

    pids_feas_df = pids_feas_df[pids_feas_df['productsku'].isin(normal_pid_list)]
    # vids_pids_sum_df = vids_pids_sum_df[vids_pids_sum_df['productsku'].isin(normal_pid_list)]

    # 存储
    vids_to_uids_df.to_csv(file_vids_to_uids, index=False)
    del vids_to_uids_df
    pids_feas_df.to_csv(file_pids_feas, index=False)
    del pids_feas_df
    uids_feas_df.to_csv(file_users_feas, index=False)
    del uids_feas_df
    gc.collect()
    vids_pids_sum_df.to_csv(file_vids_pids_sum, index=False)
    # del vids_pids_sum_df

    t3 = time.time()
    # print("预处理数据执行总花费时间： {}".format(t3 - t2))
    print("执行总花费时间： {}".format(t3 - t1))

if __name__ == '__main__':
    print("start generate predict samples ....")
    print('date: {}'.format(args.date))

    if os.path.exists(file_vids_to_uids) and os.path.exists(file_vids_pids_sum) \
        and os.path.exists(file_users_feas) and os.path.exists(file_vids_to_uids):
        print("cur_time: {} already exists!".format(cur_time))
        if args.overwrite:
            print("overwirte !")
        else:
            exit()

    hdb = hive_connect()

    # 筛选出需要 predict 的用户
    raw_data_by_sql()

    hdb.close()

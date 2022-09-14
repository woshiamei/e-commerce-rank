import time
import datetime
import pandas as pd
import os
import sys

sys.path.append("/root/reco/i2i/rank")
from utils.connect import hive_connect
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

days = 180
days_xtr = 60
# deltas = [5, 4, 3]
deltas = [i for i in range(7, 1 , -1)]

def raw_data_by_sql(delta):
    print('start sql: ')
    hdb = hive_connect()
    t1 = time.time()

    cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
    start_time = (datetime.datetime.now() - datetime.timedelta(days=days_xtr)).strftime("%Y-%m-%d").replace("-", "")
    cur_date_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
    start_date_time = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    print("cur_time: {}".format(cur_time))
    print("cur_date_time: {}".format(cur_date_time))
    print("start_time: {}".format(start_time))
    print("start_date_time: {}".format(start_date_time))

    depend_check = hdb.get_all("""
        select count(*)
        from s_bq_session_zong
        where `date` = '{}'
        and shop_mark = 'unicemall'
        and platform = '自营站'
    """.format(cur_date_time))
    #print(len(depend_check))
    if depend_check[0][0] <= 10000:
        print("count num of s_bq_session_zong is too short: {}".format(depend_check[0][0]))
        print("cur_time is {}".format(cur_date_time))
        return
        #exit()
    else:
        print("depend check done!")

    raw_sample = hdb.get_all("""
        select t5.fullvisitorid, t5.productsku, t5.email, t5.id, t5.`date`, p_view, p_click, is_addtocart, is_order
             , gender, age_range, country, region, city, register_channel, register_channel_type
             ,register_type_one, register_type_two, orders_count, totalspent, maxtotal
             ,atv_range, order_first_time, order_latest_time, orders_month, orders_3month, orders_6month
             ,orders_year, discount_num, discount_price, pay_method, `comment`, comment_num
             ,reward_point, use_reward_point, level, order_range, r, f, m, newcustomer, ordercustomer
             ,datediff_day, customer_type, customer_type_finance
             ,pro_type, product_color, product_texture, lace_size, wigtype
             ,show_sum, click_sum, add_sum, order_sum, ctr, atr, cvr
         from (
          select t3.fullvisitorid, t3.productsku, t4.id, t4.email, t3.`date`, t3.p_view, t3.p_click, t3.is_addtocart, t3.is_order,
            gender, age_range, country, region, city, register_channel, register_channel_type
            ,register_type_one, register_type_two, orders_count, totalspent, maxtotal
             ,atv_range, order_first_time, order_latest_time, orders_month, orders_3month, orders_6month
             ,orders_year, discount_num, discount_price, pay_method, `comment`, comment_num
             ,reward_point, use_reward_point, level, order_range, r, f, m, newcustomer, ordercustomer
             ,datediff_day, customer_type, customer_type_finance
          from (
                select t1.fullvisitorid, t1.`date`, t2.userid, t2.email, t1.productsku, t1.p_view, t1.p_click, t1.is_addtocart, t1.is_order
                from (
                    select fullvisitorid, productsku, p_view, p_click, is_addtocart, is_order, `date`
                    -- select count(*)
                    from ods.s_bq_custAct
                    where `date` = '{}'
                    and productsku regexp '^[0-9]+$'
                    -- and (random() < 0.5 or cast(p_click as bigint) > 0 or cast(is_addtocart as bigint) > 0 or cast(is_order as bigint) > 0)
                )t1 left join (
                    select distinct fullvisitorid, `date`, userid, email
                    -- , row_number() over(partition by fullvisitorid order by `date` desc) as rank
                    from s_bq_session_zong
                    where `date` = '{}'
                    and shop_mark = 'unicemall'
                    and platform = '自营站'
                    and email != ''
                )t2 on t1.fullvisitorid = t2.fullvisitorid
            ) t3 left join (
                select id, email,
                gender, age_range, country, region, city, register_channel, register_channel_type,
                register_type_one, register_type_two, orders_count, totalspent, maxtotal,
                atv_range, order_first_time, order_latest_time, orders_month, orders_3month, orders_6month,
                orders_year, discount_num, discount_price, pay_method, `comment`, comment_num,
                reward_point, use_reward_point, level, order_range, r, f, m, newcustomer, ordercustomer,
                datediff_day, customer_type, customer_type_finance
                from dw.dws_customer_entity
                where platfrom = "自营站"
                and shop_mark = 'unicemall'
                and brand = 'UNice'
            ) t4 on t3.email = t4.email
        )t5 left join (
            SELECT t7.product_id, pro_type, product_color, product_texture, lace_size, wigtype
                ,show_sum, click_sum, add_sum, order_sum, ctr, atr, cvr
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
                having sum(cast(p_view as BIGINT)) > 1000
            )t9 on t7.product_id = t9.productsku
        ) t6 on t5.productsku = t6.product_id
            """.format(cur_time, cur_date_time, start_date_time, start_time))
    # .format(stime))

    # print(type(user_item_action))
    print(len(raw_sample))

    t2 = time.time()
    print("sql 查询耗时： {}".format(t2 - t1))
    # print("count(cnt >= 3): {}".format(user_item_cnt))
    # for i in range(10):
    #     print(user_item_action[i])
    df = pd.DataFrame(list(raw_sample))
    df.columns = [
        'fullvisitorid', 'productsku', 'email', 'id', 'date', 'p_view', 'p_click', 'is_addtocart', 'is_order',
        'gender', 'age_range', 'country', 'region', 'city', 'register_channel', 'register_channel_type',
        'register_type_one',
        'register_type_two', 'orders_count', 'totalspent', 'maxtotal', 'atv_range', 'order_first_time',
        'order_latest_time',
        'orders_month', 'orders_3month', 'orders_6month', 'orders_year', 'discount_num', 'discount_price', 'pay_method',
        'comment', 'comment_num', 'reward_point', 'use_reward_point', 'level', 'order_range', 'r', 'f', 'm',
        'newcustomer',
        'ordercustomer', 'datediff_day', 'customer_type', 'customer_type_finance',
        'pro_type', 'product_color', 'product_texture',
        'lace_size', 'wigtype', 'show_sum', 'click_sum', 'add_sum', 'order_sum', 'ctr', 'atr', 'cvr'
    ]
    df.to_csv(raw_sample_file, index=False)

    hdb.close()

    return raw_sample

if __name__ == '__main__':
    for delta in deltas:
        cur_time = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d").replace("-", "")
        raw_sample_file = "../data/raw_sample_{}.csv".format(cur_time)
        if not os.path.exists(raw_sample_file):
            print("get data cur_time: {}".format(cur_time))
            data = raw_data_by_sql(delta)
        else:
            print("data exist: cur_time: {}".format(cur_time))
    # else:
    #     data = pd.read_csv(raw_sample_file)
    # data = raw_data_by_sql()

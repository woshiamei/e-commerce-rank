from tensorflow.keras import layers
import tensorflow as tf
#
# feature_inputs = {}
# d = {'fullvisitorid': "layers.Input((1,), name='fullvisitorid', dtype=tf.string)", 'productsku': "layers.Input((1,), name='productsku', dtype=tf.string)", 'email': "layers.Input((1,), name='email', dtype=tf.string)", 'id': "layers.Input((1,), name='id', dtype=tf.string)", 'date': "layers.Input((1,), name='date', dtype=tf.int64)", 'gender': "layers.Input((1,), name='gender', dtype=tf.string)", 'age_range': "layers.Input((1,), name='age_range', dtype=tf.string)", 'country': "layers.Input((1,), name='country', dtype=tf.string)", 'region': "layers.Input((1,), name='region', dtype=tf.string)", 'city': "layers.Input((1,), name='city', dtype=tf.string)", 'register_channel': "layers.Input((1,), name='register_channel', dtype=tf.string)", 'register_channel_type': "layers.Input((1,), name='register_channel_type', dtype=tf.string)", 'register_type_one': "layers.Input((1,), name='register_type_one', dtype=tf.string)", 'register_type_two': "layers.Input((1,), name='register_type_two', dtype=tf.string)", 'orders_count': "layers.Input((1,), name='orders_count', dtype=tf.int32)", 'totalspent': "layers.Input((1,), name='totalspent', dtype=tf.float64)", 'maxtotal': "layers.Input((1,), name='maxtotal', dtype=tf.float64)", 'atv_range': "layers.Input((1,), name='atv_range', dtype=tf.string)", 'order_first_time': "layers.Input((1,), name='order_first_time', dtype=tf.int64)", 'order_latest_time': "layers.Input((1,), name='order_latest_time', dtype=tf.int64)", 'orders_month': "layers.Input((1,), name='orders_month', dtype=tf.int32)", 'orders_3month': "layers.Input((1,), name='orders_3month', dtype=tf.int32)", 'orders_6month': "layers.Input((1,), name='orders_6month', dtype=tf.int32)", 'orders_year': "layers.Input((1,), name='orders_year', dtype=tf.int32)", 'discount_num': "layers.Input((1,), name='discount_num', dtype=tf.int32)", 'discount_price': "layers.Input((1,), name='discount_price', dtype=tf.float64)", 'pay_method': "layers.Input((1,), name='pay_method', dtype=tf.string)", 'comment': "layers.Input((1,), name='comment', dtype=tf.string)", 'comment_num': "layers.Input((1,), name='comment_num', dtype=tf.int32)", 'reward_point': "layers.Input((1,), name='reward_point', dtype=tf.float64)", 'use_reward_point': "layers.Input((1,), name='use_reward_point', dtype=tf.float64)", 'level': "layers.Input((1,), name='level', dtype=tf.string)", 'order_range': "layers.Input((1,), name='order_range', dtype=tf.string)", 'r': "layers.Input((1,), name='r', dtype=tf.string)", 'f': "layers.Input((1,), name='f', dtype=tf.string)", 'm': "layers.Input((1,), name='m', dtype=tf.string)", 'newcustomer': "layers.Input((1,), name='newcustomer', dtype=tf.string)", 'ordercustomer': "layers.Input((1,), name='ordercustomer', dtype=tf.string)", 'datediff_day': "layers.Input((1,), name='datediff_day', dtype=tf.string)", 'customer_type': "layers.Input((1,), name='customer_type', dtype=tf.string)", 'customer_type_finance': "layers.Input((1,), name='customer_type_finance', dtype=tf.string)", 'pro_type': "layers.Input((1,), name='pro_type', dtype=tf.string)", 'product_color': "layers.Input((1,), name='product_color', dtype=tf.string)", 'product_texture': "layers.Input((1,), name='product_texture', dtype=tf.string)", 'lace_size': "layers.Input((1,), name='lace_size', dtype=tf.string)", 'wigtype': "layers.Input((1,), name='wigtype', dtype=tf.string)", 'show_sum': "layers.Input((1,), name='show_sum', dtype=tf.int32)", 'click_sum': "layers.Input((1,), name='click_sum', dtype=tf.int32)", 'add_sum': "layers.Input((1,), name='add_sum', dtype=tf.int32)", 'order_sum': "layers.Input((1,), name='order_sum', dtype=tf.int32)", 'ctr': "layers.Input((1,), name='ctr', dtype=tf.float64)", 'atr': "layers.Input((1,), name='atr', dtype=tf.float64)", 'cvr': "layers.Input((1,), name='cvr', dtype=tf.float64)", 'acvr': "layers.Input((1,), name='acvr', dtype=tf.int64)"}
# for k, v in d.items():
#     print(k)
#     print(v)
#     feature_inputs[k] = v

feature_inputs = {}
feature_inputs_2 = {}
with open("./test.csv", "r") as f:
    for line in f.readlines():
        r = line.strip().split(": ")
        if len(r) != 2:
            print("error line: {}".format(line))
        na = r[0].strip("'")
        r2 = r[1].split(",")[0].split("=")[1].strip()
        s = "feature_inputs['{}'] = layers.Input((1,), name='{}', dtype={})".format(na, na, r2)
        print(s)
    #     n = s.strip().split(":")[0]
    #     m = s.strip().split(":")[1]
    #     feature_inputs[n] = m
    #     if r2 == "tf.string":
    #         feature_inputs_2[n] = layers.Input((1,), name=na, dtype=tf.string)
    #     elif r2 == "tf.int32":
    #         feature_inputs_2[n] = layers.Input((1,), name=na, dtype=tf.int32)
    #     elif r2 == "tf.int64":
    #         feature_inputs_2[n] = layers.Input((1,), name=na, dtype=tf.int64)
    #     elif r2 == "tf.float64":
    #         feature_inputs_2[n] = layers.Input((1,), name=na, dtype=tf.float64)
    # print(feature_inputs)
    # print(feature_inputs_2)


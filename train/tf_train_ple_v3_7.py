import os

import logging
import argparse
import functools

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('--dryrun', dest="dryrun", action='store_true')
parser.add_argument('--with_kai', action="store_true")
parser.add_argument('--text', action="store_true")

args = parser.parse_args()

if not args.dryrun and not args.with_kai:
    # monkey patch
    import mio_tensorflow.patch as mio_tensorflow_patch
    mio_tensorflow_patch.apply()

"""
在 cgc 基础上增加特征：user 侧， match 特征
"""
import tensorflow as tf
from mio_tensorflow.config import MioConfig
from mio_tensorflow.variable import MioEmbedding
from mio_tensorflow.layers.base import simple_dense_network, broadcast_concat, clip_to_boolean, embedding_layer, fast_dense_layer
from mio_tensorflow.layers.lhuc import simple_lhuc_network
from tensorflow.keras.backend import expand_dims,repeat_elements,sum

logging.basicConfig()

base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'base.yaml')

config = MioConfig.from_base_yaml(base_config,
                                  clear_embeddings=True, # 不要动
                                  clear_params=True, # dense 参数是否热启动
                                  # label_with_kv=True, # 不要动
                                  # dryrun=args.dryrun)
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  with_kai=args.with_kai)


training = args.mode == 'train'

@tf.custom_gradient
def sigmoid(x):
    y = tf.nn.sigmoid(x)
    def grad(dy):
      return  dy * y * (1.0 - y)
    return y, grad

def inner_layer(a, b):
    with tf.name_scope('inner_layer'):
        return tf.math.reduce_sum(a * b, keepdims=True, axis=1)

# auxiliary network
# "param.userId",
user_id = config.new_embedding("KAI_user_id", dim=32, slots=[100])
# "param.photo_id_attr",
photo_id = config.new_embedding("KAI_photo_id", dim=32, slots=[101])
# "param.author_id_attr",
author_id = config.new_embedding("KAI_author_id", dim=32, slots=[102])
# "param.type_attr",
type_attr = config.new_embedding("KAI_type_attr", dim=32, slots=[103])
# "param.hourOfDay",
hourOfDay = config.new_embedding("KAI_hourOfDay", dim=32, slots=[104])
# "param.is_friend_attr",
is_friend = config.new_embedding("KAI_is_friend", dim=32, slots=[105])
# "param.push_click_list",
push_click_list = config.new_embedding("KAI_push_click_list", dim=32, slots=[106])
# "param.push_click_type",
push_click_type = config.new_embedding("KAI_push_click_type", dim=32, slots=[107])
# "param.push_click_author",
push_click_author = config.new_embedding("KAI_push_click_author", dim=32, slots=[108])
# "param.uAgeSeg",
age_seg = config.new_embedding("KAI_age_seg", dim=4, slots=[109])
# "param.uGender",
gender = config.new_embedding("KAI_gender", dim=4, slots=[110])
# "param.uCountry",
country = config.new_embedding("KAI_country", dim=6, slots=[111])
# "param.uProvince",
province = config.new_embedding("KAI_province", dim=6, slots=[112])
# "param.relation_author_origin_list",
# relation_author_origin_list = config.new_embedding("KAI_relation_author_origin_list", dim=32, slots=[114])
# "param.push_click_cnt",
push_click_cnt = config.new_embedding("KAI_push_click_cnt", dim=8, slots=[450])
# "param.push_show_cnt",
# push_show_cnt = config.new_embedding("KAI_push_show_cnt", dim=8, slots=[116])
# "param.hist_push_ctr",
hist_push_ctr = config.new_embedding("KAI_hist_push_ctr", dim=8, slots=[451])
# "param.dAppList",
app_list = config.new_embedding("KAI_app_list", dim=32, slots=[118])
# "param.recent_follow_author",
recent_follow_author = config.new_embedding("KAI_recent_follow_author", dim=32, slots=[119])
# "param.dId",
dId = config.new_embedding("KAI_dId", dim=32, slots=[120])
# "param.push_show_list",
# push_show_list = config.new_embedding("KAI_push_show_list", dim=32, slots=[121])
# # "param.push_show_type",
# push_show_type = config.new_embedding("KAI_push_show_type", dim=32, slots=[122])
# # "param.push_show_author",
# push_show_author = config.new_embedding("KAI_push_show_author", dim=32, slots=[123])
# "param.push_click_type_x_type_attr",
push_click_type_x_type_attr = config.new_embedding("KAI_push_click_type_x_type_attr", dim=32, slots=[303])
# "param.push_show_type_x_type_attr",
# push_show_type_x_type_attr = config.new_embedding("KAI_push_show_type_x_type_attr", dim=32, slots=[307])
# "param.is_follow",
is_follow = config.new_embedding("KAI_is_follow", dim=4, slots=[402])
# app
# app_bias = config.new_embedding("KAI_app_bias", dim=4, slots=[403])
# cross
# user_id_x_type_attr = config.new_embedding("KAI_user_id_x_type_attr", dim=32, slots=[413])
# user_id_x_hourOfDay = config.new_embedding("KAI_user_id_x_hourOfDay", dim=32, slots=[414])

relation_author_origin_list_fake = config.new_embedding("KAI_relation_author_origin_list_fake", dim=16, slots=[497])
relation_author_origin_list = config.new_embedding("KAI_relation_author_origin_list", dim=16, slots=[498])
# relation_author_origin_list = config.new_embedding("KAI_relation_author_origin_list", dim=32, slots=[499])
match_type_aId = config.new_embedding("match_type", dim=2, slots=[500, 501, 502, 503])

dId_sp = config.new_embedding("KAI_dId_fm", dim=32, slots=[413])
# uId_fm = config.new_embedding("KAI_uId_fm", dim=32, slots=[431])
# pId_fm = config.new_embedding("KAI_pId_fm", dim=32, slots=[432])
# aId_fm = config.new_embedding("KAI_aId_fm", dim=32, slots=[433])
specific_ids_ks = config.new_embedding("KAI_specific_ids_ks", dim=32, slots=[431, 432, 433])
specific_ids_ne = config.new_embedding("KAI_specific_ids_ne", dim=32, slots=[434, 435, 436])

user_static_type = config.new_embedding("user_static_type", dim=4, slots=[600, 601, 602])
push_cnt_last_four_hour = config.new_embedding("KAI_push_cnt_last_four_hour", dim=2, slots=[603])

# # "ne dId",
# dId_fm = config.new_embedding("KAI_dId_fm", dim=32, slots=[420])
# uId_fm = config.new_embedding("KAI_uId_fm", dim=32, slots=[421])
# pId_fm = config.new_embedding("KAI_pId_fm", dim=32, slots=[422])
# aId_fm = config.new_embedding("KAI_aId_fm", dim=32, slots=[423])
# user_id_hot=intExtractor(['uId'], slot_id=122, is_common=True),
# author_id_hot=intExtractor(['aId'], slot_id=201),
# photo_id_hot=intExtractor(['pId'], slot_id=308),
# uId_ne = config.new_embedding("KAI_uId_ne", dim=32, slots=[415])
# uId_ne = config.new_embedding("KAI_uId_ne", dim=32, slots=[415])
# uId_ne = config.new_embedding("KAI_uId_ne", dim=32, slots=[415])
# "ks did",
# dId_ks = config.new_embedding("KAI_dId_ks", dim=32, slots=[414])
# dId_ne = config.new_embedding("KAI_dId_ne", dim=32, slots=[413])

# trainable embeddings
inputs = [
    user_id,
    photo_id,
    author_id,
    type_attr,
    hourOfDay,
    is_friend,
    push_click_list,
    push_click_type,
    push_click_author,
    age_seg,
    gender,
    country,
    province,
    relation_author_origin_list_fake,
    relation_author_origin_list,
    push_click_cnt,
    # push_show_cnt,
    hist_push_ctr,
    app_list,
    recent_follow_author,
    dId,
    # push_show_list,
    # push_show_type,
    # push_show_author,
    push_click_type_x_type_attr,
    # user_id_x_type_attr,
    # user_id_x_hourOfDay,
    # push_show_type_x_type_attr,
    # multiply特征
    tf.multiply(user_id, author_id),
    tf.multiply(push_click_author, author_id),
    tf.multiply(user_id, hourOfDay),
    tf.multiply(user_id, type_attr),
    is_follow,
    match_type_aId,
    user_static_type,
    push_cnt_last_four_hour
]

wide_inputs = config.new_embedding("wide_input32", dim=1, slots=[420, 421, 422, 423, 424, 425])
# dense_units = [256, 128, 64, 32]
out_unit = 1
batch_size = tf.shape(photo_id)[0]

# concat_input = tf.concat(inputs, axis=1)
# ctr_fm_inputs = tf.concat([dId_fm, uId_fm, pId_fm, aId_fm], axis=1)
# concat_input = tf.concat([concat_input, ctr_fm_inputs], axis=1)
wide_inputs = tf.concat(wide_inputs, axis=1)

gate_act=tf.nn.softmax
dense_units_ks = [128, 64, 16]
dense_units_ne = [128, 64, 16]
# dense_units_fm = [64, 32]
# mmoe_output = mmoe_layer(inputs, [256, 128], 64, 2, 1)

concat_input = tf.concat(inputs, axis=1)
print_ops = []
# ctr_fm_inputs = tf.concat([dId_fm, uId_fm, pId_fm, aId_fm], axis=1)
# define networks
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    # dnn = simple_dense_network(inputs, units=dense_units, name='ctr', out_unit=out_unit)
    # common 
    output = tf.layers.dense(concat_input, 256, activation=tf.nn.relu)
    output = tf.layers.dense(output, 128, activation=tf.nn.relu)
    # ks
    output_ks = tf.layers.dense(concat_input, 256, activation=tf.nn.relu)
    output_ks = tf.layers.dense(output_ks, 128, activation=tf.nn.relu)
    cur_output = tf.concat([output, output_ks], axis=1)

    gate_output_ks = fast_dense_layer(concat_input, 2, tf.nn.softmax, name='gates_ks')
    # gate_output_ks = simple_dense_network(concat_input, [], 'gate_ks', gate_act, out_unit=2)
    # gate = repeat_elements(gate_output_ks, 128, axis=1)
    # gate = add_print_node(gate, gate, "gate")
    weighted_output_ks = cur_output * repeat_elements(gate_output_ks, 128, axis=1)
    # weighted_output_ks = cur_output * gate

    # ctr_ks = simple_dense_network(weighted_output_ks, units=dense_units_ks, name='ctr_ks', out_unit=out_unit)

    # ne
    output_ne = tf.layers.dense(concat_input, 256, activation=tf.nn.relu)
    output_ne = tf.layers.dense(output_ne, 128, activation=tf.nn.relu)
    cur_output = tf.concat([output, output_ne], axis=1)

    gate_output_ne = fast_dense_layer(concat_input, 2, tf.nn.softmax, name='gates_ne')
    # gate_output_ne = simple_dense_network(concat_input, [], 'gate_ne', gate_act, out_unit=2)
    weighted_output_ne = cur_output * repeat_elements(gate_output_ne, 128, axis=1)
    
    # gate_layer_ne = simple_dense_network(concat_input, [], 'gate_ne', gate_act, out_unit=2)
    # ctr_ne = simple_dense_network(weighted_output_ne, units=dense_units_ne, name='ctr_ne', out_unit=out_unit)
    # output_ne = tf.layers.dense(weighted_output_ne, 64, activation=tf.nn.relu)
    # output_ne = tf.layers.dense(output_ne, 32, activation=tf.nn.relu)
    # ctr_ne = tf.layers.dense(output_ne, 1, activation=tf.nn.sigmoid)
    logit_ne = simple_dense_network(tf.concat([weighted_output_ne, dId_sp, specific_ids_ne], 1), units=dense_units_ne, name='ctr_nebula_3', out_unit=out_unit, out_act=None)
    logit_ks = simple_dense_network(tf.concat([weighted_output_ks, dId_sp, specific_ids_ks], 1), units=dense_units_ks, name='ctr_ks_3', out_unit=out_unit, out_act=None)
    logit_wide = tf.math.reduce_sum(wide_inputs, keepdims=True, axis=1)
    # logit_wide = 0

    ctr_ne = tf.layers.dense(logit_ne + logit_wide, 1, activation=tf.nn.sigmoid)
    ctr_ks = tf.layers.dense(logit_ks + logit_wide, 1, activation=tf.nn.sigmoid)   

    # print_ops.append(tf.print("weight:", weight, "\nlabel:", labels, "\noutput:", xtr_output, summarize=100))
    # print_ops.append(tf.print("gate_output_ne: ", gate_output_ne))
    # print_ops.append(tf.print("gate_output_ks: ", gate_output_ks))
    # print_ops.append(tf.print("gate_output_ne: ", gate_output_ne, "\ngate_output_ks", gate_output_ks, summarize=100))
    # print_ops.append(tf.print("1: {} ".format(k.get_shape().as_list())))
    # print_op.append(tf.print(ctr_fm_logit, summarize=-1))
    with tf.control_dependencies(print_ops): 
        ctr_wide = sigmoid(logit_wide)
# with tf.variable_scope('fm_model', reuse=tf.AUTO_REUSE):
#     ctr_fm = simple_dense_network(ctr_fm_inputs, units=dense_units_fm, name='ctr_fm', out_unit=out_unit)

if training:
    click = clip_to_boolean(config.get_label("click"))
    show = clip_to_boolean(config.get_label("show"))
    # type_attr = config.get_label("type_attr_index")
    app_id = config.get_label("app_id")
    server_show = tf.ones_like(click)
    zeros = tf.zeros_like(click)
    # app_id
    nebula = tf.where(tf.not_equal(app_id, 2), server_show, zeros)
    ks = tf.where(tf.not_equal(app_id, 1), server_show, zeros)
    # unknown = tf.where(tf.equal(app_id, 3), server_show, zeros)


    targets = [
        ("pctr_ne", ctr_ne, click, nebula, "auc"),
        ("pctr_ks", ctr_ks, click, ks, "auc"),
        # ("pctr_fm", ctr_fm, click, server_show, 'auc')
    ]

    # photo_type = tf.where(tf.equal(type_attr, 0), server_show, zeros)
    # live_type = tf.where(tf.equal(type_attr, 1), server_show, zeros)
    # interesting_person_type = tf.where(tf.equal(type_attr, 2), server_show, zeros)

    target_evals = [
        ("pctr_w", ctr_wide, click, server_show, "auc")
    ]
    # target_evals = [
    #     ("pctr_photo", dnn, click, photo_type, "auc"),
    #     ("pctr_live", dnn, click, live_type, "auc"),
    #     ("pctr_interest_person", dnn, click, interesting_person_type, "auc"),
    #     #("pctr_click", dnn, click, click, "auc"),
    #     #("pctr_show", dnn, click, show, "auc")     
    # ]
    
    q_names, preds, labels, weights, auc_set = zip(*targets)

    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = tf.train.GradientDescentOptimizer(1, name="opt")



    # log_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    
    # optimizer = tf.train.GradientDescentOptimizer(1, name="opt") # 别动

    # positive_label_index = tf.where(tf.equal(click,1))
    # positive_click = tf.gather_nd(click, positive_label_index)
    # positive_show = tf.gather_nd(show, positive_label_index)

    #debug_prints = [
    #    tf.print('DEBUG positive_click', positive_click, output_stream=sys.stdout),
    #    tf.print('DEBUG positive_show', positive_show, output_stream=sys.stdout)
    #]

    #with tf.control_dependencies(debug_prints):
    # opt = optimizer.minimize(log_loss)
    opt = optimizer.minimize(loss)

    q_names, preds, labels, weights, auc_set = zip(*(targets + target_evals))

    if args.dryrun:
        config.mock_and_profile([opt], './training_log/', batch_sizes=[128, 256, 512])
    elif args.with_kai:
        config.dump_kai_training_config('./training/conf', zip(q_names, preds, labels, weights, auc_set), loss=loss, text=args.text)
    else:
        config.dump_training_config('./training/conf', zip(q_names, preds, labels, weights, auc_set), opts=[opt])
else:
    targets = [
      ("pctr_ne", ctr_ne),
      ("pctr_ks", ctr_ks)
    ]
    q_names, preds = zip(*targets)
    if args.dryrun:
        config.mock_and_profile(preds, './predict_log/', batch_sizes=[200])
    else:
        config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

import time
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from utils.connect import hive_connect, BQ_Client

# 服务器没有连外网，不能查询
bq_client = BQ_Client()

create_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))

sql ="""
	select clientid, fullVisitorId, c.productSKU
	from `truemetrics-164606.116719673.ga_sessions_*`
	left join unnest(product) as c
	where   _TABLE_SUFFIX  BETWEEN '20220817' AND '20220817' and (geoNetwork.country not in('China') or geoNetwork.region not in('Shaanxi'))
		and c.productSKU <> "(not set)" and c.productSKU is not null 
		limit 10
	"""

print('query....')
bqListRet = bq_client.query(sql)
print("cid_to_vid num: {}".format(len(bqListRet)))
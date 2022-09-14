
import time
import os
from impala.dbapi import connect
from google.cloud import bigquery
AUTH_JSON_FILE_PATH = r'authenticate-15936507515.json'
# AUTH_JSON_FILE_PATH = r'truemetrics-164606-0d0cb1f9638a.json'

class BQ_Client(object):
    def __init__(self):
        # self.client = bigquery.Client.from_service_account_json('/root/reco/rank/utils/'+os.sep+AUTH_JSON_FILE_PATH)
        self.client = bigquery.Client()
        # self.client = bigquery.Client.from_service_account_json(
        #     "D:\\gaolei\\retrieval\\reco\\utils" + os.sep + AUTH_JSON_FILE_PATH)

    def query(self,sql):
        while True:
            bqList = None
            try:
                print('check in')
                bqJob = self.client.query(sql)
                print("check out")
                bqList = list(bqJob.result())  # Waits for job to complete.
            except Exception as e:
                print(e)
            if bqList:
                return bqList

class hive_connect():
    def __init__(self):
        # self.host = '192.168.1.232'
        self.host = '118.190.135.105'
        self.port = 21050
        self.conn = connect(host=self.host, port=self.port, database='ods', user='gaolei', password='gaolei.longqi')  # 连接方式
        self.cursor =self.conn.cursor()

    #关闭链接
    def close(self):
        self.cursor.close()
        self.conn.close()

    def get_one(self, sql):
        result = None
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
        except Exception as e:
            print(e)
        return result

    def get_all(self, sql):
        list = ()
        try:
            t1 = time.time()
            self.cursor.execute(sql)
            list = self.cursor.fetchall()
            t2 = time.time()
            print(t2-t1)
        except Exception as e:
            print(e)
        return list

    def get_all_to_dict(self,sql):
        """返回字典对象结果"""
        if not sql:
            return
        self.connect()
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        desc = self.cursor.description

        return [
            dict(zip([col[0] for col in desc], row))
            for row in rows
        ]


    def insert(self, sql):
        return self.__edit(sql)

    def update(self, sql):
        return self.__edit(sql)

    def upsert(self, sql):
        return self.__edit(sql)

    def delete(self, sql):
        return self.__edit(sql)

    def truncate(self,table):
        sql = """truncate table {};""".format(table)
        self.cursor.execute(sql)
        self.conn.commit()

    def __edit(self, sql):
        count = 0
        try:
            count = self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            print("error: ")
            # print(sql)
            print(e)

        return count

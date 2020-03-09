
import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/Elasticsearch_LSC_SettingUp/')
import MyLibrary_v2 as mylib
from elasticsearch import Elasticsearch
import numpy as np
import time

es = Elasticsearch([{"host": "localhost", "port": 9200}])
interest_index = "lsc2019_test_time"
server="http://localhost:9200/lsc2019_test_time/_search"

'''
import time
st_time = time.time()
result = search_es_sequence_of_2(info1, info2, max_len=150)
ed_time = time.time()
print(f"Exc Time: {ed_time - st_time} seconds")
'''

for idx in range(10):
    st_time = time.time()
    sent = 'after watch tv at 6pm, supermarket, then use laptop'
    final = mylib.search_es_temporal(sent, ES=es, index=interest_index, mins_between_events=10)
    #final = mylib.search_es_temporal_server_api(sent, server=server, mins_between_events=10)
    ed_time = time.time()
    print(f"{len(final)} -- Exc Time: {ed_time - st_time} seconds")

#final
import json
import numpy as np
from operator import itemgetter
from collections import Counter
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import random
import time
from dateutil.parser import parse
import Tag_Event as tage
import ProcImgLib as imglib 
import pickle
import requests

stop_words = stopwords.words('english')
stop_words += [',', '.']
ps = PorterStemmer()

Datapath = "/Users/duynguyen/DuyNguyen/Gitkraken/Elasticsearch_LSC_SettingUp/data"

###### Load Group ########
Group_file = Datapath + "/Grouping_Info.pickle"
with open(Group_file, "rb") as f:
    list_group = pickle.load(f)
list_group_str = [','.join(x) for x in list_group]
list_mean_time_group = [imglib.mean_time_stamp(x)[1] for x in list_group]

###### Load Synonym ########
Synonym_glove_all_file = Datapath + "/List_synonym_glove_all.pickle"
with open(Synonym_glove_all_file, "rb") as f:
    list_synonym = pickle.load(f)

##### Load Synonym stemmed #####
Synonym_file = Datapath + '/List_synonym_glove_all_stemmed.pickle'
with open(Synonym_file, 'rb') as f:
    list_synonym_stemmed = pickle.load(f)

##### Load dictionary #### --> Also feature vector format
with open(Datapath + '/bow_my_dictionary.pickle', "rb") as f:
    my_dictionary = pickle.load(f)

def create_weekday_query(list_weekday=['Saturday'], field='weekday', boost=24):
    if len(list_weekday) > 0:
        weekday_part = " OR ".join(list_weekday)
        stt = True
        weekday_json = create_json_query_string_part(query=weekday_part, field=field, boost=boost)
    else:
        stt = False
        weekday_json = 0
    return stt, weekday_json

# ====== SEQUENTIAL MOMENTS =========
def generate_query_text(list_queries, list_filters, tie_breaker=0.7, numb_of_result=100):
    '''
    See previous version
    Add list_filters ==> machenism is changed ==> using bool + filter if list_filter exist
    '''

    dismax_part = "{\"dis_max\":{\"queries\":["

    queries_part_string = str(list_queries)
    queries_part_string = queries_part_string.replace("'", "\"")
    queries_part_string = queries_part_string.replace("doc[\"description_embedded\"]", "doc['description_embedded']")
    queries_part_string = queries_part_string[1:-1]
    dismax_part += queries_part_string
    dismax_part += "],\"tie_breaker\":" + str(tie_breaker) + "}"
    dismax_part += "}"

    result = "{\"size\":" + str(numb_of_result)
    result += ",\"_source\": {\"includes\":" 
    result += "[\"id\", \"description\", \"time\", \"group\"]" + "}"
    result += ",\"query\":{\"bool\":{\"must\":["  
    result += dismax_part + "]"

    if len(list_filters) > 0:
        result += ",\"filter\":" + str(list_filters).replace("'","\"")
    
    result += "}" + "}" + "}"
    return result

def create_location_query(list_loc=['home'], field='address', boost=24):
    if len(list_loc) > 0:
        location_part = " OR ".join(list_loc)
        stt = True
        location_json = create_json_query_string_part(query=location_part, field=field, boost=boost)
    else:
        stt = False
        location_json = 0
    return stt, location_json

def convert_text_to_date(t):
    result = ''
    if '/' in t: # in the right format (should be dd/mm/yyyy, can miss yyyy part):
        part = t.split('/')
        for idx, x in enumerate(part):
            if len(x) == 1:
                part[idx] = f'0{x}'
            else:
                part[idx] = x
        result = '/'.join(part)
        if len(part) == 2: # miss the year
            result += '/2016'
    else: # in case data only mentioned about month (without day)
        result = parse(t)
        result = result.month
        if result < 10:
            result = f"0{result}"
        else:
            result = str(result)
    return result

def create_json_query_string_part(query, field, boost=1):
    if query != "" and query != " " and len(query) >= 2:
        result = {
            "query_string":{
                "query": query,
                "default_field": field,
                "boost": boost
            }
        }
    else:
        result = None
    return result
    
def generate_list_dismax_part_and_filter_time_from_info(info):
    # info is a dict with keys of obj, loc, period, time, timeofday
    # Time will be in filter --> No score
    field_clip = 'description_clip'
    field_des = 'description'
    delta_hour = 0.5

    # For obj and loc --> query as normal
    obj_sentence = ', '.join(info['obj'])
    loc_sentence = ', '.join(info['loc'])

    queries_part = []
    basic_obj_part = create_json_query_string_part(query=obj_sentence, field=field_clip, boost=3)
    loc_as_obj_part = create_json_query_string_part(query=loc_sentence, field=field_clip, boost=3)
    loc_as_obj_part_des = create_json_query_string_part(query=loc_sentence, field=field_des, boost=5)
    
    queries_part = queries_part + [basic_obj_part] if basic_obj_part is not None else queries_part
    queries_part = queries_part + [loc_as_obj_part] if loc_as_obj_part is not None else queries_part
    queries_part = queries_part + [loc_as_obj_part_des] if loc_as_obj_part_des is not None else queries_part
    
    having_location, location_query = create_location_query(info['loc'], field="address", boost=10)
    queries_part = queries_part + [location_query] if having_location else queries_part
    
    # Expand object, location here


    # For time and timeofday --> filter is better
    filters_part = []
    if len(info['time']) > 0:
        for time in info['time']:
            if time in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                time_json = [{ "match":  { "weekday": time }}]
            else:
                time_convert = convert_text_to_date(time)
                if len(time_convert) == 10: # full day, and month, year (be added)
                    day = int(time_convert[0:2])
                    month = int(time_convert[3:5])
                else: # only month
                    day = -1
                    month = int(time_convert)
                time_json = [{"term":{"month": month}}]
                if day >= 0:
                    time_json += [{"term": {"day": day}}]
            filters_part += time_json
    
    if len(info['timeofday']) > 0:
        time_json = []
        for time in info['timeofday']:
            if ';' in time:
                temp = time.split('; ')
                oclock = parse(temp[1])
                oclock = oclock.hour
                if temp[0] in ['after']:
                    time_json += [{"range" : {"hour" : {"gte" : oclock-delta_hour}}}]
                elif temp[0] in ['to', 'til', 'before']:
                    time_json += [{"range" : {"hour" : {"lte" : oclock+delta_hour}}}]
                elif temp[0] in ['at', 'around']:
                    time_json += [{"range" : {"hour" : {"gte": oclock-delta_hour, "lte" : oclock+delta_hour}}}]
            else:
                oclock = parse(time)
                oclock = oclock.hour
                time_json += [{"range" : {"hour" : {"gte" : oclock-1}}}]
        filters_part += time_json
        
    return queries_part, filters_part

def group_list_images_and_calculate_score_detail(id_list, sc_list, time_delta=100):
    # Group images into groups (SIFT and Description) and calculate average score of that group
    # time_delta (int) maximum minutes to be grouped --> if later or sooner than this thershold --> new group
    # Rank descending
    # Output is List of [group(list), score(scalar)]
   
    id_all = list(set(id_list))
    id_list_array = np.asarray(id_list)
    sc_list_array = np.asarray(sc_list)
    group_id = imglib.grouping_image_with_sift_dict(sorted(id_all), time_delta=time_delta)
    score_group_id = []
    for group in group_id:
        score_id = 0
        for id_img in group:
            idx = np.where(id_list_array == id_img)[0]
            sc_id = np.sum(sc_list_array[idx])
            score_id += sc_id
        score_id /= len(group)
        score_group_id.append(score_id)
    
    sorted_score_index = sorted(range(len(score_group_id)), key=lambda k: score_group_id[k], reverse=True)
    sc_result = sorted(score_group_id, reverse=True)
    id_result = [group_id[x] for x in sorted_score_index]
    final = [[x, y] for x, y in zip(id_result, sc_result)]
    return final

def group_list_images_and_calculate_score(id_list, sc_list, group_list):
    # Rank descending
    # Output is List of [group(list), score(scalar), group_result(string)]
    # group result is string of id name ','.join
    id_all = list(set(id_list))
    unique_group_list_str = list(set(group_list))
    group_id = []
    score_group_id = []

    for unique_group in unique_group_list_str:
        unique_group_list = unique_group.split(',')
        id_img_in_group = [x for x in id_all if x in unique_group_list]
        group_id.append(id_img_in_group)
        score_id = [sc_list[x] for x in range(len(id_list)) if id_list[x] in unique_group_list]
        score_id = np.asarray(score_id)
        score_id = np.sum(score_id) / len(id_img_in_group)
        score_group_id.append(score_id)
    
    sorted_score_index = sorted(range(len(score_group_id)), key=lambda k: score_group_id[k], reverse=True)
    sc_result = sorted(score_group_id, reverse=True)
    id_result = [group_id[x] for x in sorted_score_index]
    group_result = [unique_group_list_str[x] for x in sorted_score_index]
    final = [[x, y, z] for x, y, z in zip(id_result, sc_result, group_result)]
    return final

def expend_time_from_previous_result(group_result, time_after=2):
    # time_after can be negative informing previous action

    time_group = []
    range_time = []
    time_after_datetime = imglib.datetime.timedelta(hours=time_after)

    for x in group_result:
        time, time_datetime = imglib.mean_time_stamp(x)
        Flag = True
        for y in time_group:
            time_delta = abs(time_datetime - y)
            if time_delta < time_after_datetime:
                Flag = False
                break
        if Flag:
            time_group.append(time_datetime)
            time_extend = time_datetime + time_after_datetime
            time_extend_str = time_extend.strftime('%Y%m%d_%H%M%S')
            if time_datetime < time_extend:
                range_query = {
                    'range':{
                        "time":{
                            "gte": f"{time[0:4]}/{time[4:6]}/{time[6:8]} {time[9:11]}:{time[11:13]}:{time[13:15]}",
                            "lte": f"{time_extend_str[0:4]}/{time_extend_str[4:6]}/{time_extend_str[6:8]} {time_extend_str[9:11]}:{time_extend_str[11:13]}:{time_extend_str[13:15]}",
                            "boost": 3
                        }
                    }
                }
            else:
                range_query = {
                    'range':{
                        "time":{
                            "lte": f"{time[0:4]}/{time[4:6]}/{time[6:8]} {time[9:11]}:{time[11:13]}:{time[13:15]}",
                            "gte": f"{time_extend_str[0:4]}/{time_extend_str[4:6]}/{time_extend_str[6:8]} {time_extend_str[9:11]}:{time_extend_str[11:13]}:{time_extend_str[13:15]}",
                            "boost": 3
                        }
                    }
                }
            range_time += [range_query]
    return time_group, range_time

# ========== SERVER API SEARCH FUNCTION ==========
def search_es_server_api(server, request, percent_thres=0.9, max_len=10):
    # Input: 
    # - server is the link the elastic index server (e.g: http://localhost:9200/lsc2019_test_time/_search)
    # - request is json format to search in elastic
    headers = {"Content-Type": "application/json"}
    res = requests.post(server, headers=headers, data=request)
    
    res = res.json()
    numb_result = len(res["hits"]["hits"])
    print("Total searched result: " + str(numb_result))

    if numb_result != 0:
        score = [d["_score"] for d in res["hits"]["hits"]]  # Score of all result (higher mean better)
        id_result = [d["_source"]["id"] for d in res["hits"]["hits"]]  # List all id images in the result
        group_result = [d["_source"]["group"] for d in res["hits"]["hits"]]

        score_np = np.asarray(score)

        max_score = score_np[0]
        thres_score = max_score * percent_thres

        index_filter = np.where(score_np > thres_score)[0]

        if len(index_filter) > 1:
            result = list(itemgetter(*list(index_filter))(id_result))
            result_score = list(itemgetter(*list(index_filter))(score))
            group_result = list(itemgetter(*list(index_filter))(group_result))
            numb_result = len(result)
        else:
            result = id_result[0]
            result_score = score[0]
            group_result = group_result[0]
            numb_result = 1

        if numb_result > max_len:
            result = result[0:max_len]
            result_score = result_score[0:max_len]
            group_result = group_result[0:max_len]

        #print("Total remaining result: " + str(numb_result))  # Number of result
        #print("Total output result: " + str(min(max_len, numb_result)))  # Number of result
    else:
        result_score = []
        result = []
        group_result = []
        
    return result_score, result, group_result

def search_es_sequence_of_2_server_api(info1, info2, server, max_len=100, time_after = 2,):
    # Search es for sequence info1 --> info2
    # time_after is hours that info2 happens after info1
    # group image in each infor within time_after/2 hour * 60 minutes
    # Pass forward and backward to increase accuracy
    # search info1 --> result1 --> search info2 after result1 --> update result1 --> update result2
    time_after_datetime = imglib.datetime.timedelta(hours=time_after)

    query_1, filter_1 = generate_list_dismax_part_and_filter_time_from_info(info1)
    query_2, filter_2 = generate_list_dismax_part_and_filter_time_from_info(info2)

    # forward 1
    es_query_1 = generate_query_text(query_1, filter_1)
    score_result_1_1, id_result_1_1, group_result_1_1 = search_es_server_api(server=server, request=es_query_1, 
                                        percent_thres = 0.5, max_len=max_len)

    es_query_2 = generate_query_text(query_2, filter_2)
    score_query_2_1, id_result_2_1, group_result_2_1 = search_es_server_api(server=server, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)
    
    if len(id_result_1_1) > 0 and len(id_result_2_1) > 0:
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, group_result_1_1)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2 = expend_time_from_previous_result(group_result_1, time_after)
        query_2_added = query_2 + range_time_for_query_2
        es_query_2 = generate_query_text(query_2_added, filter_2)
        score_result_2_1, id_result_2_1, group_result_2_1 = search_es_server_api(server=server, request=es_query_2, 
                                            percent_thres = 0.5, max_len=max_len)

        
        # backward 1
        group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, group_result_2_1)
        group_result_2 = [x[0] for x in group_result_2_with_score]
        time_group_2, range_time_for_query_1 = expend_time_from_previous_result(group_result_2, time_after=-time_after)
        query_1_added = query_1 + range_time_for_query_1
        es_query_1 = generate_query_text(query_1_added, filter_1)
        score_result_1_2, id_result_1_2, group_result_1_2 = search_es_server_api(server=server, request=es_query_1, 
                                            percent_thres = 0.5, max_len=max_len)

        # Forward 2
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_2, score_result_1_2, group_result_1_2)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2 = expend_time_from_previous_result(group_result_1, time_after)
        query_2_added = query_2 + range_time_for_query_2
        es_query_2 = generate_query_text(query_2_added, filter_2)
        score_result_2_2, id_result_2_2, group_result_2_2 = search_es_server_api(server=server, request=es_query_2, 
                                            percent_thres = 0.5, max_len=max_len)
        
        score_result = [score_result_1_1, score_result_1_2, score_result_2_1, score_result_2_2]
        id_result = [id_result_1_1, id_result_1_2, id_result_2_1, id_result_2_2]
        group_result = [group_result_1_1, group_result_1_2, group_result_2_1, group_result_2_2]

        # Group sequence actions into final return
        group_2 = group_list_images_and_calculate_score(id_result[2] + id_result[3], score_result[2] + score_result[3], group_result[2] + group_result[3])
        score_2 = [x[1] for x in group_2]
        idx_2 = [x[0] for x in group_2]
        list_time_string_2 = [x[2] for x in group_2]
        group_1 = group_list_images_and_calculate_score(id_result[0] + id_result[1], score_result[0] + score_result[1], group_result[0] + group_result[1])
        score_1 = [x[1] for x in group_1]
        idx_1 = [x[0] for x in group_1]
        list_time_string_1 = [x[2] for x in group_1]

        mean_time_1 = [imglib.mean_time_stamp(x)[1] for x in idx_1] # datetime type
        result = []

        for idx2, group_2 in enumerate(idx_2):
            time_2 = imglib.mean_time_stamp(group_2)[1]
            list_idx_1_satisfy_2= [idx_x for idx_x in range(len(mean_time_1)) if abs(time_2 - mean_time_1[idx_x]) < time_after_datetime and time_2 > mean_time_1[idx_x]]
            time_string_2 = list_time_string_2[idx2]

            # Looking whether find action 1 within the time limit of current action 2
            if len(list_idx_1_satisfy_2) != 0:
                for idx_1_st_2 in list_idx_1_satisfy_2:
                    score_group = (2*score_2[idx2] + score_1[idx_1_st_2])/3 # weighted following action have higher score than the previous
                    result.append([idx_1[idx_1_st_2], group_2, score_group, list_time_string_1[idx_1_st_2], time_string_2])
        
        # Sort score again
        score_2_group = [x[2] for x in result]
        sorted_score_index = sorted(range(len(score_2_group)), key=lambda k: score_2_group[k], reverse=True)
        final = [result[x] for x in sorted_score_index]            

    else: # It should not happen this cases (if it does --> need fix!)
        if len(id_result_1_1) > 0:
            final = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, group_result_1_1)
        elif len(id_result_2_1) > 0:
            final = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, group_result_2_1)
        else:
            final = []
    return final

def search_es_sequence_of_3_server_api(info1, info2, info3, server, max_len=100, time_after=2):
    #time_after = 2
    #max_len = 100

    time_after_datetime = imglib.datetime.timedelta(hours=time_after)

    query_1, filter_1 = generate_list_dismax_part_and_filter_time_from_info(info1)
    query_2, filter_2 = generate_list_dismax_part_and_filter_time_from_info(info2)
    query_3, filter_3 = generate_list_dismax_part_and_filter_time_from_info(info3)

    es_query_1 = generate_query_text(query_1, filter_1, numb_of_result=max_len)
    score_result_1_1, id_result_1_1, group_result_1_1 = search_es_server_api(server=server, request=es_query_1, 
                                        percent_thres = 0.5, max_len=max_len)

    # es_query_2 = mylib.generate_query_text(query_2, filter_2)
    # score_query_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
    #                                     percent_thres = 0.5, max_len=max_len)

    es_query_3 = generate_query_text(query_3, filter_3, numb_of_result=max_len)
    score_result_3_1, id_result_3_1, group_result_3_1 = search_es_server_api(server=server, request=es_query_3, 
                                        percent_thres = 0.5, max_len=max_len)

    if len(id_result_1_1) > 0:
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, group_result_1_1)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2_from_1 = expend_time_from_previous_result(group_result_1, time_after=time_after)
    else:
        range_time_for_query_2_from_1 = []
    if len(id_result_3_1) > 0:
        group_result_3_with_score = group_list_images_and_calculate_score(id_result_3_1, score_result_3_1, group_result_3_1)
        group_result_3 = [x[0] for x in group_result_3_with_score]
        time_group_3, range_time_for_query_2_from_3 = expend_time_from_previous_result(group_result_3, time_after=-time_after)
    else:
        range_time_for_query_2_from_3 = []

    # es_query_2 = mylib.generate_query_text(query_2, filter_2)
    # score_result_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
    #                                     percent_thres = 0.5, max_len=max_len)
    # init_group_result_2 = imglib.grouping_image_with_sift_dict(sorted(id_result_2_1))

    # Update time for query 2
    query_2_added = query_2 + range_time_for_query_2_from_1 + range_time_for_query_2_from_3
    es_query_2 = generate_query_text(query_2_added, filter_2, numb_of_result=max_len)
    score_result_2_1, id_result_2_1, group_result_2_1 = search_es_server_api(server=server, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)

    # Update time for query 1 and 3
    group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, group_result_2_1)
    group_result_2 = [x[0] for x in group_result_2_with_score]
    _, range_time_for_query_1_from_2 = expend_time_from_previous_result(group_result_2, time_after=-time_after)
    _, range_time_for_query_3_from_2 = expend_time_from_previous_result(group_result_2, time_after=time_after)

    query_1_added = query_1 + range_time_for_query_1_from_2
    query_3_added = query_3 + range_time_for_query_3_from_2
    es_query_1 = generate_query_text(query_1_added, filter_1, numb_of_result=max_len)
    score_result_1_2, id_result_1_2, group_result_1_2 = search_es_server_api(server=server, request=es_query_1, 
                                                      percent_thres = 0.5, max_len=max_len)

    es_query_3 = generate_query_text(query_3_added, filter_3, numb_of_result=max_len)
    score_result_3_2, id_result_3_2, group_result_3_2 = search_es_server_api(server=server, request=es_query_3, 
                                                      percent_thres = 0.5, max_len=max_len)

    # Update time for query 2
    group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_2, score_result_1_2, group_result_1_2)
    group_result_1 = [x[0] for x in group_result_1_with_score]
    time_group_1, range_time_for_query_2_from_1 = expend_time_from_previous_result(group_result_1, time_after=time_after)

    group_result_3_with_score = group_list_images_and_calculate_score(id_result_3_2, score_result_3_2, group_result_3_2)
    group_result_3 = [x[0] for x in group_result_3_with_score]
    time_group_3, range_time_for_query_2_from_3 = expend_time_from_previous_result(group_result_3, time_after=-time_after)

    query_2_added = query_2 + range_time_for_query_2_from_1 + range_time_for_query_2_from_3
    es_query_2 = generate_query_text(query_2_added, filter_2, numb_of_result=max_len)
    score_result_2_2, id_result_2_2, group_result_2_2 = search_es_server_api(server=server, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)
    group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_2, score_result_2_2, group_result_2_2)

    # Group sequence action into final return
    score_1 = [x[1] for x in group_result_1_with_score]
    idx_1 = group_result_1
    score_2 = [x[1] for x in group_result_2_with_score]
    idx_2 = [x[0] for x in group_result_2_with_score]
    score_3 = [x[1] for x in group_result_3_with_score]
    idx_3 = group_result_3

    mean_time_1 = [imglib.mean_time_stamp(x)[1] for x in idx_1] # datetime type
    mean_time_3 = [imglib.mean_time_stamp(x)[1] for x in idx_3] # datetime type
    result = []

    for idx2, group_2 in enumerate(idx_2):
        time_2 = imglib.mean_time_stamp(group_2)[1]
        list_idx_1_satisfy_2= [idx_x for idx_x in range(len(mean_time_1)) if abs(time_2 - mean_time_1[idx_x]) < time_after_datetime and time_2 > mean_time_1[idx_x]]
        list_idx_3_satisfy_2 = [idx_x for idx_x in range(len(mean_time_3)) if abs(time_2 - mean_time_3[idx_x]) < time_after_datetime and time_2 < mean_time_3[idx_x]]
        
        # Looking whether find action 1 and action 3 within the time limit of current action 2
        if len(list_idx_1_satisfy_2) * len(list_idx_3_satisfy_2) != 0:
            for idx_1_st_2 in list_idx_1_satisfy_2:
                for idx_3_st_2 in list_idx_3_satisfy_2:
                    score_group = (2*score_2[idx2] + score_1[idx_1_st_2] + score_3[idx_3_st_2])/4 # weighted following action have higher score than the previous
                    result.append([sorted(idx_1[idx_1_st_2]), sorted(group_2), sorted(idx_3[idx_3_st_2]), score_group])
     # Sort score again
    score_groups = [x[2] for x in result]
    sorted_score_index = sorted(range(len(score_groups)), key=lambda k: score_groups[k], reverse=True)
    final = [result[x] for x in sorted_score_index]    

    return final

#sent = 'flower, vase, old clock'
#sent = 'after watching tv, went to dcu by car, then used computer in the office'
#sent = 'after watching tv, went to dcu by car'

def search_es_temporal_server_api(sent, server="http://localhost:9200/lsc2019_test_time/_search", mins_between_events=10):
    list_tense = ['past', 'present', 'future']
    time_between_events = imglib.datetime.timedelta(hours=mins_between_events/60) # 10 minutes

    info_full = tage.extract_info_from_sentence_full_tag(sent)
    contain_info = [1 if len(info_full[x]) > 0 else 0 for x in info_full] # check which tense contains information

    if sum(contain_info) == 1:
        max_len = 600 # since only 1 infomation and will be grouped --> more images to have more groups to view
        idx = contain_info.index(1)
        tense = list_tense[idx]
        info = info_full[tense]
        query_part, filter_part = generate_list_dismax_part_and_filter_time_from_info(info)
        es_query = generate_query_text(query_part, filter_part)
        score_result, id_result, group_result = search_es_server_api(server=server, request=es_query, percent_thres = 0.5, max_len=max_len)
        # Group result and rank based on average score
        result = group_list_images_and_calculate_score(id_result, score_result, group_result)

        current_event = [x[0] for x in result]
        score_event = [x[1] for x in result]
        idx_event_from_list = [list_group_str.index(x[2]) for x in result]
        # Find previous and following event
        previous_event = []
        following_event = []
        for idx_event in idx_event_from_list:
            for idx_previous in range(idx_event-1, -2, -1):
                if list_mean_time_group[idx_event] - list_mean_time_group[idx_previous] > time_between_events:
                    previous_event.append(list_group[idx_previous])
                    break
                if idx_previous == -1:
                    previous_event.append(list_group[0])
                    break
            for idx_following in range(idx_event+1, len(list_mean_time_group)+1):
                if list_mean_time_group[idx_following] - list_mean_time_group[idx_event] > time_between_events:
                    following_event.append(list_group[idx_following])
                    break
                if idx_previous == len(list_mean_time_group):
                    following_event.append(list_group[-1])
                    break
        final = [[sorted(x), sorted(y), sorted(z), t] for x, y, z, t in zip(previous_event, current_event, following_event, score_event)]
    elif sum(contain_info) == 2:
        max_len = 450
        time_after = 2
        idx_discard = contain_info.index(0)
        idx = [x for x in range(3) if x != idx_discard]
        info1 = info_full[list_tense[idx[0]]]
        info2 = info_full[list_tense[idx[1]]]
        result = search_es_sequence_of_2_server_api(info1, info2, server=server, max_len=max_len, time_after=time_after)
        
        if len(result) == 0:
            final = []
            return final

        score_event = [x[2] for x in result]

        if idx_discard == 0: # missing previous action
            previous_event = []
            current_event = [x[0] for x in result]
            following_event = [x[1] for x in result]
            idx_event_from_list = [list_group_str.index(x[3]) for x in result]
            for idx_event in idx_event_from_list:
                for idx_previous in range(idx_event-1, -2, -1):
                    if list_mean_time_group[idx_event] - list_mean_time_group[idx_previous] > time_between_events:
                        previous_event.append(list_group[idx_previous])
                        break
                    if idx_previous == -1:
                        previous_event.append(list_group[0])
                        break

        else: # missing following action (idx_discard = 2) --> idx_discard should not be 1 (miss current event)
            following_event = []
            current_event = [x[1] for x in result]
            previous_event = [x[0] for x in result]
            idx_event_from_list = [list_group_str.index(x[4]) for x in result]
            for idx_event in idx_event_from_list:
                for idx_following in range(idx_event+1, len(list_mean_time_group)+1):
                    if list_mean_time_group[idx_following] - list_mean_time_group[idx_event] > time_between_events:
                        following_event.append(list_group[idx_following])
                        break
                    if idx_following == len(list_mean_time_group):
                        following_event.append(list_group[-1])
                        break
        
        final = [[sorted(x), sorted(y), sorted(z), t] for x, y, z, t in zip(previous_event, current_event, following_event, score_event)]

    elif sum(contain_info) == 3: # appear past, present, future action
        max_len = 300
        time_after = 2
        info1 = info_full['past']   
        info2 = info_full['present']
        info3 = info_full['future']
        final = search_es_sequence_of_3_server_api(info1, info2, info3, server=server, max_len=max_len, time_after=time_after)
    else: # no info detected
        final = []
    return final


            

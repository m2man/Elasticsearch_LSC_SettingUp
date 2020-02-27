
import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/Elasticsearch_LSC_SettingUp/')
import src.Tag_Event as tage
import src.MyLibrary_v2 as mylib
import src.ProcImgLib as imglib 
from elasticsearch import Elasticsearch
import numpy as np

es = Elasticsearch([{"host": "localhost", "port": 9200}])
interest_index = "lsc2019_test_time"

sent = 'after watching tv at 6am, went to DCU by car, then used my computer at the helix'
info_full = tage.extract_info_from_sentence_full_tag(sent)
info1 = info_full['past']
info2 = info_full['present']
info3 = info_full['future']

#past_query, past_filter = mylib.generate_list_dismax_part_from_info(past_info)
#present_query, present_filter = mylib.generate_list_dismax_part_from_info(present_info)
#future_query, future_filter = mylib.generate_list_dismax_part_from_info(future_info)

def group_list_images_and_calculate_score(id_list, sc_list, time_delta=100):
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

def expend_time_from_previous_result(group_result, time_after=2):
    time_group = []
    range_time = []
    for x in group_result:
        Flag = False
        time, time_datetime = imglib.mean_time_stamp(x)
        try:
            time_previous_datetime = time_group[-1]
        except:
            time_previous_datetime = None
        if time_previous_datetime is None:
            time_group.append(time_datetime)
            Flag = True
        else:
            time_delta = abs(time_datetime - time_previous_datetime)
            if time_delta >= abs(imglib.datetime.timedelta(hours=time_after)):
                time_group.append(time_datetime)
                Flag = True
        if Flag:
            time_extend = time_datetime + imglib.datetime.timedelta(hours=time_after)
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

def search_es_sequence_of_2(info1, info2, max_len=100, time_after = 2):
    # Search es for sequence info1 --> info2
    # time_after is hours that info2 happens after info1
    # group image in each infor within time_after/2 hour * 60 minutes
    # Pass forward and backward to increase accuracy
    # search info1 --> result1 --> search info2 after result1 --> update result1 --> update result2
    time_after_datetime = imglib.datetime.timedelta(hours=time_after)

    query_1, filter_1 = mylib.generate_list_dismax_part_and_filter_time_from_info(info1)
    query_2, filter_2 = mylib.generate_list_dismax_part_and_filter_time_from_info(info2)

    # forward 1
    es_query_1 = mylib.generate_query_text(query_1, filter_1)
    score_result_1_1, id_result_1_1 = mylib.search_es(es, index=interest_index, request=es_query_1, 
                                        percent_thres = 0.5, max_len=max_len)

    es_query_2 = mylib.generate_query_text(query_2, filter_2)
    score_query_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)
    
    if len(id_result_1_1) > 0 and len(id_result_2_1) > 0:
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, time_delta=time_after*30)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2 = expend_time_from_previous_result(group_result_1, time_after)
        query_2_added = query_2 + range_time_for_query_2
        es_query_2 = mylib.generate_query_text(query_2_added, filter_2)
        score_result_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
                                            percent_thres = 0.5, max_len=max_len)

        
        # backward 1
        group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, time_delta=time_after*30)
        group_result_2 = [x[0] for x in group_result_2_with_score]
        time_group_2, range_time_for_query_1 = expend_time_from_previous_result(group_result_2, time_after=-time_after)
        query_1_added = query_1 + range_time_for_query_1
        es_query_1 = mylib.generate_query_text(query_1_added, filter_1)
        score_result_1_2, id_result_1_2 = mylib.search_es(es, index=interest_index, request=es_query_1, 
                                            percent_thres = 0.5, max_len=max_len)

        # Forward 2
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_2, score_result_1_2, time_delta=time_after*30)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2 = expend_time_from_previous_result(group_result_1, time_after)
        query_2_added = query_2 + range_time_for_query_2
        es_query_2 = mylib.generate_query_text(query_2_added, filter_2)
        score_result_2_2, id_result_2_2 = mylib.search_es(es, index=interest_index, request=es_query_2, 
                                            percent_thres = 0.5, max_len=max_len)
        
        score_result = [score_result_1_1, score_result_1_2, score_result_2_1, score_result_2_2]
        id_result = [id_result_1_1, id_result_1_2, id_result_2_1, id_result_2_2]

        # Group sequence actions into final return
        group_2 = group_list_images_and_calculate_score(id_result[2] + id_result[3], score_result[2] + score_result[3], time_delta=time_after*30)
        score_2 = [x[1] for x in group_2]
        idx_2 = [x[0] for x in group_2]
        group_1 = group_list_images_and_calculate_score(id_result[0] + id_result[1], score_result[0] + score_result[1], time_delta=time_after*30)
        score_1 = [x[1] for x in group_1]
        idx_1 = [x[0] for x in group_1]


        mean_time_1 = [imglib.mean_time_stamp(x)[1] for x in idx_1] # datetime type
        result = []

        for idx2, group_2 in enumerate(idx_2):
            time_2 = imglib.mean_time_stamp(group_2)[1]
            list_idx_1_satisfy_2= [idx_x for idx_x in range(len(mean_time_1)) if abs(time_2 - mean_time_1[idx_x]) < time_after_datetime]
            
            # Looking whether find action 1 within the time limit of current action 2
            if len(list_idx_1_satisfy_2) != 0:
                for idx_1_st_2 in list_idx_1_satisfy_2:
                    score_group = (2*score_2[idx2] + score_1[idx_1_st_2])/3 # weighted following action have higher score than the previous
                    result.append([idx_1[idx_1_st_2], group_2, score_group])
        
        # Sort score again
        score_2_group = [x[2] for x in result]
        sorted_score_index = sorted(range(len(score_2_group)), key=lambda k: score_2_group[k], reverse=True)
        final = [result[x] for x in sorted_score_index]            

    else:
        if len(id_result_1_1) > 0:
            final = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, time_delta=60)
        elif len(id_result_2_1) > 0:
            final = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, time_delta=60)
        else:
            final = []
    return final

def search_es_sequence_of_3(info1, info2, info3, max_len=100, time_after=2):
    #time_after = 2
    #max_len = 100

    time_after_datetime = imglib.datetime.timedelta(hours=time_after)

    query_1, filter_1 = mylib.generate_list_dismax_part_and_filter_time_from_info(info1)
    query_2, filter_2 = mylib.generate_list_dismax_part_and_filter_time_from_info(info2)
    query_3, filter_3 = mylib.generate_list_dismax_part_and_filter_time_from_info(info3)

    es_query_1 = mylib.generate_query_text(query_1, filter_1)
    score_result_1_1, id_result_1_1 = mylib.search_es(es, index=interest_index, request=es_query_1, 
                                        percent_thres = 0.5, max_len=max_len)

    # es_query_2 = mylib.generate_query_text(query_2, filter_2)
    # score_query_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
    #                                     percent_thres = 0.5, max_len=max_len)

    es_query_3 = mylib.generate_query_text(query_3, filter_3)
    score_result_3_1, id_result_3_1 = mylib.search_es(es, index=interest_index, request=es_query_3, 
                                        percent_thres = 0.5, max_len=max_len)

    if len(id_result_1_1) > 0:
        group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_1, score_result_1_1, time_delta=time_after*30)
        group_result_1 = [x[0] for x in group_result_1_with_score]
        time_group_1, range_time_for_query_2_from_1 = expend_time_from_previous_result(group_result_1, time_after=time_after)
    else:
        range_time_for_query_2_from_1 = []
    if len(id_result_3_1) > 0:
        group_result_3_with_score = group_list_images_and_calculate_score(id_result_3_1, score_result_3_1, time_delta=time_after*30)
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
    es_query_2 = mylib.generate_query_text(query_2_added, filter_2)
    score_result_2_1, id_result_2_1 = mylib.search_es(es, index=interest_index, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)

    # Update time for query 1 and 3
    group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_1, score_result_2_1, time_delta=time_after*30)
    group_result_2 = [x[0] for x in group_result_2_with_score]
    _, range_time_for_query_1_from_2 = expend_time_from_previous_result(group_result_2, time_after=-time_after)
    _, range_time_for_query_3_from_2 = expend_time_from_previous_result(group_result_2, time_after=time_after)

    query_1_added = query_1 + range_time_for_query_1_from_2
    query_3_added = query_3 + range_time_for_query_3_from_2
    es_query_1 = mylib.generate_query_text(query_1_added, filter_1)
    score_result_1_2, id_result_1_2 = mylib.search_es(es, index=interest_index, request=es_query_1, 
                                                      percent_thres = 0.5, max_len=max_len)

    es_query_3 = mylib.generate_query_text(query_3_added, filter_3)
    score_result_3_2, id_result_3_2 = mylib.search_es(es, index=interest_index, request=es_query_3, 
                                                      percent_thres = 0.5, max_len=max_len)

    # Update time for query 2
    group_result_1_with_score = group_list_images_and_calculate_score(id_result_1_2, score_result_1_2, time_delta=time_after*30)
    group_result_1 = [x[0] for x in group_result_1_with_score]
    time_group_1, range_time_for_query_2_from_1 = expend_time_from_previous_result(group_result_1, time_after=time_after)

    group_result_3_with_score = group_list_images_and_calculate_score(id_result_3_2, score_result_3_2, time_delta=time_after*30)
    group_result_3 = [x[0] for x in group_result_3_with_score]
    time_group_3, range_time_for_query_2_from_3 = expend_time_from_previous_result(group_result_3, time_after=-time_after)

    query_2_added = query_2 + range_time_for_query_2_from_1 + range_time_for_query_2_from_3
    es_query_2 = mylib.generate_query_text(query_2_added, filter_2)
    score_result_2_2, id_result_2_2 = mylib.search_es(es, index=interest_index, request=es_query_2, 
                                        percent_thres = 0.5, max_len=max_len)
    group_result_2_with_score = group_list_images_and_calculate_score(id_result_2_2, score_result_2_2, time_delta=time_after*30)

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
        list_idx_1_satisfy_2= [idx_x for idx_x in range(len(mean_time_1)) if abs(time_2 - mean_time_1[idx_x]) < time_after_datetime]
        list_idx_3_satisfy_2 = [idx_x for idx_x in range(len(mean_time_3)) if abs(time_2 - mean_time_3[idx_x]) < time_after_datetime]
        
        # Looking whether find action 1 and action 3 within the time limit of current action 2
        if len(list_idx_1_satisfy_2) * len(list_idx_3_satisfy_2) != 0:
            for idx_1_st_2 in list_idx_1_satisfy_2:
                for idx_3_st_2 in list_idx_3_satisfy_2:
                    score_group = (2*score_2[idx2] + score_1[idx_1_st_2] + score_3[idx_3_st_2])/4 # weighted following action have higher score than the previous
                    result.append([idx_1[idx_1_st_2], group_2, idx_3[idx_3_st_2], score_group])
     # Sort score again
    score_groups = [x[2] for x in result]
    sorted_score_index = sorted(range(len(score_groups)), key=lambda k: score_groups[k], reverse=True)
    final = [result[x] for x in sorted_score_index]    

    return final



    


# def group_list_images_and_calculate_score(list_id, list_score):
#     # Group images into groups (SIFT and Description) and calculate average score of that group
#     # Rank descending
#     # Output is List of [group(list), score(scalar)]
#     score = []
#     group_result = imglib.grouping_image_with_sift_dict(sorted(list_id))
#     for group in group_result:
#         group_score = 0
#         for x in group:
#             group_score += list_score[list_id.index(x)]
#         group_score /= len(group)
#         score.append(group_score)
#     sorted_score_index = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
#     sorted_group_result = [group_result[x] for x in sorted_score_index]
#     score = sorted(score, reverse=True)
#     final = [[x, y] for x, y in zip(sorted_group_result, score)]
#     return final

'''
import time
st_time = time.time()
result = search_es_sequence_of_2(info1, info2, max_len=150)
ed_time = time.time()
print(f"Exc Time: {ed_time - st_time} seconds")
'''
#sent = 'after a flight, walking through an airbridge'
list_tense = ['past', 'present', 'future']
sent = 'flower, vase, old clock'

info_full = tage.extract_info_from_sentence_full_tag(sent)
contain_info = [1 if len(info_full[x]) > 0 else 0 for x in info_full] # check which tense contains information

if sum(contain_info) == 1:
    max_len = 400 # since only 1 infomation and will be grouped --> more images to have more groups to view
    idx = contain_info.index(1)
    tense = list_tense[idx]
    info = info_full[tense]
    query_part, filter_part = mylib.generate_list_dismax_part_and_filter_time_from_info(info)
    es_query = mylib.generate_query_text(query_part, filter_part)
    score_result, id_result = mylib.search_es(es, index=interest_index, request=es_query, percent_thres = 0.5, max_len=max_len)
    # Group result and rank based on average score
    final = group_list_images_and_calculate_score(id_result, score_result)
elif sum(contain_info) == 2:
    max_len = 150
    time_after = 2
    idx_discard = contain_info.index(0)
    idx = [x for x in range(3) if x != idx_discard]
    info1 = contain_info[list_tense[idx[0]]]
    info2 = contain_info[list_tense[idx[1]]]
    final = search_es_sequence_of_2(info1, info2, max_len=max_len, time_after=time_after)
elif sum(contain_info) == 3: # appear past, present, future action
    max_len = 100
    time_after = 2
    info1 = info_full['past']   
    info2 = info_full['present']
    info3 = info_full['future']
    final = search_es_sequence_of_3(info1, info2, info3, max_len=max_len, time_after=time_after)
else: # no info detected
    final = []




#result = search_es_sequence_of_2(info_full['past'], info_full['present'], max_len=150)


'''
# ======== OLD VERSION =======
def combine_result_with_score_v0(id1, sc1, id2, sc2, max_len=100):
    id_share = [x for x in id1 if x in id2]
    sc_share = [sc1[id1.index(x)] + sc2[id2.index(x)] for x in id_share]
    index_sorted_sc_share = sorted(range(len(sc_share)), key=lambda k: sc_share[k], reverse=True)
    sorted_id_share = [id_share[x] for x in index_sorted_sc_share]
    sorted_sc_share = [sc_share[x] for x in index_sorted_sc_share]

    id1_not_in_id2 = [x for x in id1 if x not in id_share]
    sc1_not_in_id2 = [sc1[id1.index(x)] for x in id1_not_in_id2]
    id2_not_in_id1 = [x for x in id2 if x not in id_share]
    sc2_not_in_id1 = [sc2[id2.index(x)] for x in id2_not_in_id1]
    id_dist = id1_not_in_id2 + id2_not_in_id1
    sc_dist = sc1_not_in_id2 + sc2_not_in_id1
    index_sorted_sc_dist = sorted(range(len(sc_dist)), key=lambda k: sc_dist[k], reverse=True)
    sorted_id_dist = [id_dist[x] for x in index_sorted_sc_dist]
    sorted_sc_dist = [sc_dist[x] for x in index_sorted_sc_dist]

    id_result = sorted_id_share + sorted_id_dist
    sc_result = sorted_sc_share + sorted_sc_dist
    if len(id_result) > max_len:
        id_result = id_result[0:max_len]
        sc_result = sc_result[0:max_len]
    return sc_result, id_result
    
'''


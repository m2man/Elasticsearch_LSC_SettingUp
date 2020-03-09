import os
import json
from pathlib import Path
import cv2
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

os.chdir("/Users/duynguyen/DuyNguyen/Gitkraken/Elasticsearch_LSC_SettingUp/")
Data_path = 'data'
Image_path = 'LSC_DATA/'

description_file = Data_path + '/description_all.json'
with open(description_file) as json_file:
    description = json.load(json_file)


with open(Data_path + '/sift_dict_jpg.pickle', 'rb') as f:
    sift_dict = pickle.load(f)


list_img = list(description.keys())
sorted_list = sorted(list_img)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def match_images(img1_ft, img2_ft):
    matches = bf.match(img1_ft, img2_ft)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def find_folder_image_from_name(a):
    if len(a) > 8:
        a = a[0:8]
    folder = a[0:4] + '-' + a[4:6] + '-' + a[6:8] 
    return folder

def convert_to_time(a):
    info = a.split('_')
    result = datetime.datetime(int(info[0][0:4]), int(info[0][4:6]), int(info[0][6:8]), 
                                int(info[1][0:2]), int(info[1][2:4]), int(info[1][4:6]))
    return result

def concate_description(a):
    a_scene = a['scene_image']
    a_obj = a['object_image']
    a_obj_list = a_obj.split(', ')
    for idx, val in enumerate(a_obj_list):
        t = val.split()
        a_obj_list[idx] = t[-1]
    a_description = a_scene + ', ' + ', '.join(a_obj_list)
    return a_description

def description_between_images(img1_name, img2_name, description):
    img1_description = description[img1_name]
    img2_description = description[img2_name]

    img1_description = concate_description(img1_description)
    img2_description = concate_description(img2_description)

    img1_list = img1_description.split(', ')
    img2_list = img2_description.split(', ')
    share = [x for x in img1_list if x in img2_list]

    share_ratio = 2*len(share) / (len(img1_list) + len(img2_list))

    return share_ratio

def mean_time_stamp(list_string_time):
    sum_time = 0
    for a in list_string_time:
        a = a[0:15]
        a_time = convert_to_time(a)
        a_timestamp = datetime.datetime.timestamp(a_time)
        sum_time += a_timestamp
    sum_time /= len(list_string_time)
    mean_time = datetime.datetime.fromtimestamp(sum_time)
    str_mean_time = mean_time.strftime('%Y%m%d_%H%M%S')
    return str_mean_time, mean_time

# Create SIFT Ft dict
def create_sift_dict(description):
    list_images = list(description.keys())
    sorted_list = sorted(list_images)

    sift_dict = {}

    for idx in tqdm(range(len(sorted_list))):
        img = sorted_list[idx]
        img = img[0:-4]+'.png'
        img1 = cv2.imread(Image_path + '/' + find_folder_image_from_name(img) + '/' + img, 0)
        kp1, des1 = orb.detectAndCompute(img1, None)
        if len(kp1) == 0:
            des1 = None
        sift_dict[img] = des1
    
    with open(Data_path+'/sift_dict.pickle', 'wb') as f:
        pickle.dump(sift_dict, f)

#reate_sift_dict(description)

def grouping_image_with_sift_dict(list_images, description=description, sift_dict=sift_dict, time_delta = 100):
    # time_delta (int) maximum minutes to be grouped --> if later or sooner than this thershold --> new group
    topk = 10
    slope1 = 0.7
    slope2 = 0.3
    thres = 1.7
    time_break = datetime.timedelta(minutes = time_delta)
    
    result = []
    n_images = len(list_images)
    idx_start = 0
    idx_end = 0
    while idx_start < n_images and idx_end < n_images - 1:
        name_img1 = list_images[idx_start]
        time_img1 = convert_to_time(name_img1)
        sift_ft_1 = sift_dict[name_img1]
        idx_end = idx_start
        group = [name_img1]
        flag_group = True
        while flag_group and idx_end < n_images - 1:
            idx_end += 1
            name_img2 = list_images[idx_end]
            time_img2 = convert_to_time(name_img2)
            time_delta = abs(time_img1 - time_img2)
            if time_delta > time_break:
                flag_group = False
            else:
                sift_ft_2 = sift_dict[name_img2]
                if sift_ft_2 is not None:
                    distance_img = match_images(sift_ft_1, sift_ft_2)
                    distance_img = [x.distance for x in distance_img]
                    distance_img = np.mean(distance_img[0:min(topk, len(distance_img))])
                    distance_des = 1 - description_between_images(name_img2, name_img1, description)
                    distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                    if distance < thres:
                        group.append(name_img2)
                        sift_ft_1 = sift_ft_2
                        name_img1 = name_img2
                    elif idx_end < n_images - 1:
                        name_img3 = list_images[idx_end+1]
                        time_img3 = convert_to_time(name_img3)
                        time_delta = abs(time_img3 - time_img2)
                        if time_delta < datetime.timedelta(seconds=60):
                            sift_ft_3 = sift_dict[name_img3]
                            if sift_ft_3 is not None:
                                distance_img = match_images(sift_ft_1, sift_ft_3)
                                distance_img = [x.distance for x in distance_img]
                                distance_img = np.mean(distance_img[0:min(topk, len(distance_img))])
                                distance_des = 1 - description_between_images(name_img3, name_img1, description)
                                distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                                if distance < thres:
                                    group.append(name_img2)
                                    group.append(name_img3)
                                    sift_ft_1 = sift_ft_3
                                    name_img1 = name_img3
                                    idx_end += 1
                                else:
                                    flag_group = False
                                    idx_end = idx_end
                        else:
                            result.append([name_img2])
                            flag_group = False
                            idx_end = idx_end + 1
                    else:
                        flag_group = False
                        result.append([name_img2])
                        idx_end = n_images-1
                    
        result.append(group)                
        idx_start = idx_end

    return result

'''
list_images = list(description.keys())
sorted_list = sorted(list_images)
initial_result = sorted_list[10:40] + sorted_list[100:120] + sorted_list[555: 570] + sorted_list[1555: 1580]
a = grouping_image_with_sift_dict(initial_result, description, sift_dict)

#JPEG error: 20160901_234356_000
'''

'''
# ======= OLD VERSION ========
def grouping_image_with_sift_dict_v0(list_images, description, sift_dict):
    topk = 10
    slope1 = 0.7
    slope2 = 0.3
    thres = 1.7

    time_break = datetime.timedelta(minutes = 100)
    result = []
    n_images = len(list_images)
    idx_process = 0
    idx_start = 0
    while idx_process < n_images and idx_start < n_images - 1:
        name_img1 = list_images[idx_process]
        time_img1 = convert_to_time(name_img1)
        sift_ft_1 = sift_dict[name_img1]
        idx_start = idx_process
        group = [name_img1]
        flag_group = True
        while flag_group and idx_start < n_images - 1:
            idx_start += 1
            name_img2 = list_images[idx_start]
            time_img2 = convert_to_time(name_img2)
            time_delta = abs(time_img1 - time_img2)
            if time_delta > time_break:
                idx_process = idx_start - 1
                flag_group = False
            else:
                sift_ft_2 = sift_dict[name_img2]
                if sift_ft_2 is not None:
                    distance_img = match_images(sift_ft_1, sift_ft_2)
                    distance_img = [x.distance for x in distance_img]
                    distance_img = np.mean(distance_img[0:min(topk, len(distance_img))])
                    distance_des = 1 - description_between_images(name_img2, name_img1, description)
                    distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                    if distance < thres:
                        group.append(name_img2)
                    else:
                        name_img3 = list_images[idx_start+1]
                        time_img3 = convert_to_time(name_img3)
                        time_delta = abs(time_img3 - time_img2)
                        if time_delta < datetime.timedelta(seconds=60):
                            sift_ft_3 = sift_dict[name_img3]
                            if sift_ft_3 is not None:
                                distance_img = match_images(sift_ft_1, sift_ft_3)
                                distance_img = [x.distance for x in distance_img]
                                distance_img = np.mean(distance_img[0:min(topk, len(distance_img))])
                                distance_des = 1 - description_between_images(name_img3, name_img1, description)
                                distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                                if distance < thres:
                                    group.append(name_img2)
                                    group.append(name_img3)
                                    idx_start += 1
                                else:
                                    idx_process = idx_start - 1
                                    flag_group = False
                            else:
                                idx_process = idx_start + 1
                                flag_group = False
                        else:
                            idx_process = idx_start
                            result.append([name_img2])
                            flag_group = False        
                else:
                    if time_delta < datetime.timedelta(seconds=90):
                        group.append(name_img2)
                    else:
                        idx_process = idx_start
                        flag_group = False

        result.append([group])                
        idx_process += 1

    return result


'''

# list_images = list(description.keys())
# sorted_list = sorted(list_images)

# print('Processing ...')
# group = grouping_image_with_sift_dict(sorted_list)
# print('Finished!')
# with open('Grouping_Info.pickle', 'wb') as f:
#     pickle.dump(group, f)
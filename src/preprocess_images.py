'''
import os
import shutil
import glob
import os.path
from os import path

folders = ['LSC_DATA 2', 'LSC_DATA 3', 'LSC_DATA 4', 'LSC_DATA 5']
workingdir = os.getcwd() + '/LSCDATA'

for k in folders:
    print(f"Processing {k}")
    from_folder = os.listdir(workingdir + '/' + k + '/')
    for f_folder in from_folder:
        if f_folder != 'resources':
            allfiles = os.listdir(workingdir + '/' + k + '/' + f_folder + '/')
            for f_file in allfiles:
                folder_exist = path.exists(workingdir + '/' + 'LSC_DATA' + '/' + f_folder)
                if folder_exist == False:
                    os.makedirs(workingdir + '/' + 'LSC_DATA' + '/' + f_folder)
                shutil.move(workingdir + '/' + k + '/' + f_folder + '/' + f_file, workingdir + '/' + 'LSC_DATA' + '/' + f_folder + '/' + f_file)
'''

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

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def convert_to_time(a):
    info = a.split('_')
    result = datetime.datetime(int(info[0][0:4]), int(info[0][4:6]), int(info[0][6:8]), 
                                int(info[1][0:2]), int(info[1][2:4]), int(info[1][4:6]))
    return result

def find_folder_image_from_name(a):
    if len(a) > 8:
        a = a[0:8]
    folder = a[0:4] + '-' + a[4:6] + '-' + a[6:8] 
    return folder

def match_images(img1_ft, img2_ft):
    matches = bf.match(img1_ft, img2_ft)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def distance_between_images(img1_name, img2_name, topk=50, plot=False):
    # Using ORB (similar to SIFT-SURF) to find match keypoints and calculate mean distance between them
    # Smaller mean more similar
    img1_name = img1_name.replace('jpg','png')
    img2_name = img2_name.replace('jpg','png')
    img1 = cv2.imread(Image_path + '/' + find_folder_image_from_name(img1_name) + '/' + img1_name, 0)
    kp1, des1 = orb.detectAndCompute(img1, None)
    img2 = cv2.imread(Image_path + '/' + find_folder_image_from_name(img2_name) + '/' + img2_name, 0)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if len(kp1) == 0  or len(kp2) == 0:
        mean_distance = 0
    else:
        m = match_images(des1, des2)

        distance = [x.distance for x in m]
        mean_distance = np.mean(distance[0:min(topk, len(distance))])

        if plot:
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,m[0:topk], None, flags=2)
            plt.imshow(img3),plt.show()

    return mean_distance

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

def group_sequence_action(description, min_time_dif=5, n_images=40):
    list_images = list(description.keys())
    sorted_list = sorted(list_images)      

    slope1 = 0.7
    slope2 = 0.3
    thres = 1.72

    test_present = sorted_list[0:n_images]
    test_past = ['a' for _ in range(len(test_present))]
    test_future = ['a' for _ in range(len(test_present))]

    time_delta = datetime.timedelta(minutes=min_time_dif)

    for idx_present in tqdm(range(len(test_present))):
        name_present = test_present[idx_present]
        time_present = convert_to_time(name_present)
        
        idx_past = idx_present
        idx_future = idx_present
        flag_past = False
        flag_future = False

        if test_past[idx_present] == 'a':
            while flag_past == False:
                idx_past = idx_past - 1
                if idx_past < 0 or idx_present - idx_past >= 250:
                    idx_past = max(-1, idx_past)
                    flag_past = True
                else:
                    name_past = test_present[idx_past]
                    time_past = convert_to_time(name_past)
                    dif_time = time_present - time_past
                    dif_day = time_present.day - time_past.day
                    if dif_day > 0:
                        flag_past = True
                        idx_past = -1
                    else:
                        if dif_time > time_delta:
                            distance_img = distance_between_images(name_present, name_past, topk=10, plot=False)
                            distance_des = 1 - description_between_images(name_present, name_past, description)
                            distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                            if distance > thres:
                                flag_past = True

        while flag_future == False:
            idx_future = idx_future + 1
            if idx_future >= len(test_present) or idx_future - idx_present >= 250:
                flag_future = True
                idx_future = min(len(test_present) - 1, idx_future)
            else:
                name_future = test_present[idx_future]
                time_future = convert_to_time(name_future)
                dif_time = time_future - time_present
                dif_day = time_future.day - time_present.day
                if dif_day > 0:
                    flag_future = True
                    idx_future = -1
                else:
                    if dif_time > time_delta:
                        distance_img = distance_between_images(name_present, name_future, topk=10, plot=False)
                        distance_des = 1 - description_between_images(name_present, name_future, description)
                        distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                        if distance > thres:
                            flag_future = True
        
        if idx_past >= 0:
            test_past[idx_present] = test_present[idx_past]
        else:
            test_past[idx_present] = ''

        if idx_future >= 0:
            test_future[idx_present] = test_present[idx_future]
            test_past[idx_future] = test_present[idx_present]
        else:
            test_future[idx_present] = ''

    return test_past, test_present, test_future

def group_sequence_action_faster(description, min_time_dif=5, n_images=40):
    list_images = list(description.keys())
    sorted_list = sorted(list_images)      

    slope1 = 0.7
    slope2 = 0.3
    thres = 1.7

    test_present = sorted_list[0:n_images]
    test_past = ['a' for _ in range(len(test_present))]
    test_future = ['a' for _ in range(len(test_present))]
    time_delta = datetime.timedelta(minutes=min_time_dif)

    idx_future = 0
    idx_past = -1
    last = -1
    til = -1
    last_ft = 0
    til_ft = 0

    for idx_present in tqdm(range(len(test_present))):
        #name_present = test_present[idx_present]
        #time_present = convert_to_time(name_present)
        
        if last_ft <= 0:
            flag_future = False
        else:
            if idx_present < last_ft:
                flag_future = True
            else:
                last_ft = til_ft
                flag_future = False

        while flag_future == False:
            til_ft += 1
            if til_ft >= len(test_present) or til_ft - last_ft >= 250:
                til_ft = min(len(test_present)-1, til_ft)
                flag_future = True
                if last_ft == 0:
                    last_ft = til_ft
                last = til
                til = idx_present
                idx_past = int((last+til)/2) - 1
            else:
                name_last_ft = test_present[last_ft]
                time_last_ft = convert_to_time(name_last_ft)
                name_future = test_present[til_ft]
                time_future = convert_to_time(name_future)
                dif_time = time_future - time_last_ft
                dif_day = time_future.day - time_last_ft.day
                if dif_day > 0:
                    flag_future = True
                    if last_ft == 0:
                        last_ft = til_ft
                    last = til
                    til = idx_present
                    idx_past = int((last+til)/2) - 1

                if dif_time > time_delta and dif_day == 0:
                    distance_img = distance_between_images(name_last_ft, name_future, topk=10, plot=False)
                    distance_des = 1 - description_between_images(name_last_ft, name_future, description)
                    distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                    if distance > thres:
                        if last_ft == 0:
                            last_ft = til_ft
                        flag_future = True
                        last = til
                        til = idx_present
                        idx_past = int((last+til)/2) - 1

        if til_ft == last_ft:
            flag_future = False
            while flag_future == False:
                til_ft += 1
                if til_ft >= len(test_present) or til_ft - last_ft >= 250:
                    til_ft = min(len(test_present)-1, til_ft)
                    flag_future = True
                else:
                    name_last_ft = test_present[last_ft]
                    time_last_ft = convert_to_time(name_last_ft)
                    name_future = test_present[til_ft]
                    time_future = convert_to_time(name_future)
                    dif_time = time_future - time_last_ft
                    if dif_time > time_delta:
                        distance_img = distance_between_images(name_last_ft, name_future, topk=10, plot=False)
                        distance_des = 1 - description_between_images(name_last_ft, name_future, description)
                        distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1+slope2)
                        if distance > thres:
                            flag_future = True

        idx_future = int((last_ft + til_ft)/2)

        if idx_past >= 0:
            test_past[idx_present] = test_present[idx_past]
        else:
            test_past[idx_present] = ''

        if idx_future >= 0:
            test_future[idx_present] = test_present[idx_future]
        else:
            test_future[idx_present] = ''

        #print(idx_past, idx_present, idx_future, last_ft, til_ft)

    return test_past, test_present, test_future

def plot_sequence_action(past_list, present_list, future_list, idx=0):
    present = present_list[idx]
    past = past_list[idx]
    future = future_list[idx]

    if past != '':
        img1 = cv2.imread(Image_path + '/' + find_folder_image_from_name(past) + '/' + past)
        name1 = f"Past:{past[:-8]}"
    else:
        img1 = None
        name1 = 'Past: None'
    img2 = cv2.imread(Image_path + '/' + find_folder_image_from_name(present) + '/' + present)
    name2 = present[:-8]
    if future != '':
        img3 = cv2.imread(Image_path + '/' + find_folder_image_from_name(future) + '/' + future)
        name3 = f"Future: {future[:-8]}"
    else:
        img3 = None
        name3 = 'Future: None'

    plot_3_images(img1, img2, img3, name1, name2, name3)

def plot_3_images(im1, im2, im3, n1='', n2='', n3=''):
    fig = plt.figure(figsize=(10,3))
    plt1 = fig.add_subplot(131)
    plt2 = fig.add_subplot(132)
    plt3 = fig.add_subplot(133)

    if im1 is not None:
        plt1.imshow(im1)
        plt1.set_title(n1)

    if im2 is not None:
        plt2.imshow(im2)
        plt2.set_title(n2)

    if im3 is not None:
        plt3.imshow(im3)
        plt3.set_title(n3)

list_images = list(description.keys())
sorted_list = sorted(list_images)

#sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

slope1 = 0.7
slope2 = 0.3
thres = 1.72

'''
img1_name = sorted_list[0]
img2_name = sorted_list[1]
distance_img = distance_between_images(img1_name, img2_name, topk=10, plot=False)
distance_des = 1 - description_between_images(img1_name, img2_name, description)

mean_distance = 2*(slope1*distance_img/35 + slope2*distance_des)/(slope1 + slope2)

if mean_distance <= thres:
    print('Match')
else:
    print("NOT Match")
'''

def post_processing(past, present, future):
    duplicate_past = sorted(list_duplicates(past))
    for idx in range(len(duplicate_past)):
        img = duplicate_past[idx][0]
        list_dup = duplicate_past[idx][1]
        list_dup = [x for x in range(list_dup[0], list_dup[-1]+1)]
        if img != '':
            for idx_dup in list_dup:
                future[idx_dup] = future[list_dup[-1]]

    duplicate_future = sorted(list_duplicates(future))
    for idx in range(len(duplicate_future)):
        img = duplicate_future[idx][0]
        list_dup = duplicate_future[idx][1]
        list_dup = [x for x in range(list_dup[0], list_dup[-1]+1)]
        if img != '':
            for idx_dup in list_dup:
                past[idx_dup] = past[list_dup[0]]

    result = {}
    result['past'] = past
    result['present'] = present
    result['future'] = future

    return result

def run_grouping(description, time_delta):
    past, present, future = group_sequence_action_faster(description, min_time_dif= time_delta, n_images=len(description))
    result = {}
    result['past'] = past
    result['present'] = present
    result['future'] = future
    with open(f"Sequence_{time_delta}_prior_faster.pickle","wb") as f:
        pickle.dump(result, f)

    result = post_processing(past, present, future)
    with open(f"Sequence_{time_delta}_post_faster.pickle","wb") as f:
        pickle.dump(result, f)


timedif = [5]
#timedif = [1]
for time_delta in timedif:
    print(f"Processing {time_delta} ... ")
    run_grouping(description, time_delta=time_delta)


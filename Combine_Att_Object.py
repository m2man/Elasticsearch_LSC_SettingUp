#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:49:16 2019
Read 2 preprocessed json file (object and attribute) then merge them into 1 json file
Perform Stem on the text data (optional)
Create small description with only small amount of images for testing
@author: duyphd
"""

import json
from nltk.stem import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize 
from random import random
from tqdm import tqdm

def stemming_sentence(s):
    '''
    stemming all word in a sentence s
    Output --> result is the stemed sentence
    '''
    ps = PorterStemmer()
    # ps = EnglishStemmer()
    result = ''
    words = word_tokenize(s)
    for word in words:
        result = result + ps.stem(word) + ' '
    result = result[:-1]
    return result

##### Read json file #####
print("Loading json ...")

File = '/Volumes/GoogleDrive/My Drive/BackUp/LSC2019_Test/attribute_all.json'

with open(File) as json_file:
    data_att = json.load(json_file)
    
File = '/Volumes/GoogleDrive/My Drive/BackUp/LSC2019_Test/object_yolo_all.json'

with open(File) as json_file:
    data_obj = json.load(json_file)
    
File = '/Volumes/GoogleDrive/My Drive/BackUp/LSC2019_Test/object_yolo_all_extend.json'

with open(File) as json_file:
    data_obj_format = json.load(json_file)
    
##### Perform merging #####
id_image_att = list(data_att.keys())
id_image_obj = list(data_obj.keys())
id_image_obj_format = list(data_obj_format.keys())

check = [x in id_image_obj for x in id_image_att] # if sum(check) = len(id_image_obj) --> all obj is in att
assert sum(check) == len(id_image_obj)
# Result checking: object miss 1 image (41662 images while attribute has 41663 images)

# Merge 2 list --> only keep unique value
id_image_all = id_image_att + id_image_obj
id_image_all = list(set(id_image_all))

description = {}
description_small = {}
n_small_threshold = 2000
n_count = 0

for i in tqdm(range(len(id_image_all))):
    id_image = id_image_all[i]
    
#    print('Process: ' + id_image)
    
    scence_image = ''
    object_image = 'Nothing'
    object_format_image = "Nothing"
    
    if id_image in id_image_att:    
        scene_image = data_att[id_image]
    if id_image in id_image_obj:    
        object_image = data_obj[id_image]
    if id_image in id_image_obj_format:    
        object_format_image = data_obj_format[id_image]
        
    #scence_image_stem = stemming_sentence(scence_image)
    #object_image_stem = stemming_sentence(object_image)
    #object_format_image_stem = stemming_sentence(object_format_image)
    
    description[id_image] = {
        "scene_image": scene_image,
        "object_image": object_image,
        "object_format_image": object_format_image
    }
    
    if n_count < n_small_threshold:
        if random() > 0.4:
            description_small[id_image] = {
                "scene_image": scene_image,
                "object_image": object_image,
                "object_format_image": object_format_image
            }
            n_count += 1
    
with open('description_all.json', 'w') as outfile:
    json.dump(description, outfile)
    
with open('description_small.json', 'w') as outfile:
    json.dump(description_small, outfile)
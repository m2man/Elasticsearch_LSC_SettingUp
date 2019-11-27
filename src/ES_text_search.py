#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:33:34 2019
Create Elasticsearch database locally and add document from combined descriptione json to the database
Then perform simple query
@author: duyphd
"""

import json
import pickle
from elasticsearch import Elasticsearch
from tqdm import tqdm
#import time
#import MyLibrary as mylib
import json
#from PIL import Image
from datetime import datetime
import calendar

###### Load list synonym ########
#Synonym_glove_object_all_file = "/Volumes/GoogleDrive/Msy Drive/BackUp/LSC2019_Test/List_synonym_glove_object_yolo_all.pickle"
#with open(Synonym_glove_object_all_file, "rb") as f:
#    List_synonym_glove_object_yolo_all = pickle.load(f)
#
#Synonym_glove_att_all_file = "/Volumes/GoogleDrive/My Drive/BackUp/LSC2019_Test/List_synonym_glove_att_all.pickle"
#with open(Synonym_glove_att_all_file, "rb") as f:
#    List_synonym_glove_att_all = pickle.load(f)
#
#List_synonym = List_synonym_glove_att_all + List_synonym_glove_object_yolo_all

Synonym_glove_all_file = "List_synonym_glove_all.pickle"
with open(Synonym_glove_all_file, "rb") as f:
    List_synonym = pickle.load(f)


####### Connect to the elastic cluster --> run elasticsearch first #########
es = Elasticsearch([{"host": "localhost", "port": 9200}])

interest_index = "lsc2019"

try:
    es.indices.delete(index=interest_index)
except Exception:
    print("Do not have index to delete: " + interest_index)

########### Add analyzer for the client ##########
es.indices.create(
    index=interest_index,
    body={
            "settings": {
                    # just one shard, no replicas for testing
                    "number_of_shards": 1,
                    "number_of_replicas": 0,

                    # custom analyzer for analyzing file paths
                    "analysis": {
                            # Set up analyer --> include 
                            #   + Char_Filter(useless)
                            #   + Tokenizer (token word)
                            #   + Filter (tokenizer filter: filter for tokenized token) --> stem, synonym
                            "analyzer": { 
                                    "analyzer_tfidf": { # Set up analyzer first --> then define own custom thing later
                                            "type": "custom",
                                            "tokenizer": "tokenizer_tfidf",
                                            "filter":[
                                                    "english_possessive_stemmer",
                                                    "lowercase",
                                                    "english_stop",
                                                    "english_keywords",
                                                    "english_stemmer"
                                            ]
                                    },
                                    "analyzer_search":{ # Define another analyzer for search (usually the same with analyzer of index, but now we will do different)
                                            "type": "custom",
                                            "tokenizer": "standard", # Just simply lower the query search
                                            "filter": [
                                                    "my_graph_synonym", # Synonym filter
                                                    "lowercase",
                                                    "english_stop",
                                                    "english_keywords",
                                                    "english_possessive_stemmer",
                                                    "english_stemmer"
                                            ]
                                    },
                                    "analyzer_object_yolo_term":{
                                            "type": "custom",
                                            "tokenizer": "tokenizer_term", # remove 'and' and ',' 
                                            "filter": [
                                                    "lowercase",
                                                    "english_stop",
                                                    "english_keywords",
                                                    "english_possessive_stemmer",
                                                    "english_stemmer"
                                            ]
                                    },
                                    "analyzer_object_yolo_term_clip":{
                                            "type": "custom",
                                            "tokenizer": "standard", # remove 'and' and ',' 
                                            "filter": [
                                                    "lowercase",
                                                    "english_stop",
                                                    "english_keywords",
                                                    "english_possessive_stemmer",
                                                    "english_stemmer"
                                            ]
                                    },
                                    "analyzer_object_yolo_term_clip_search":{
                                            "type": "custom",
                                            "tokenizer": "standard", # remove 'and' and ',' 
                                            "filter": [
                                                    "my_graph_synonym",
                                                    "lowercase",
                                                    "english_stop",
                                                    "english_keywords",
                                                    "english_possessive_stemmer",
                                                    "english_stemmer"
                                            ]
                                    }
                             },
                            "tokenizer":{
                                    "tokenizer_tfidf":{
                                            "type": "edge_ngram",
                                            "min_gram": 1,
                                            "max_gram": 10,
                                            "token_chars":[
                                                    "letter",
                                                    "digit"
                                             ]
                                     },
                                    "tokenizer_term":{                                            
                                            "type": "simple_pattern_split",
                                            "pattern": "(( and )|(, ))"
                                    }
                             },
                            "filter": {
                                    "english_stop": {
                                      "type":       "stop",
                                      "stopwords":  "_english_" 
                                    },
                                    "english_keywords": {
                                      "type":       "keyword_marker",
                                      "keywords":   ["example"] 
                                    },
                                    "english_stemmer": {
                                      "type":       "stemmer",
                                      "language":   "english"
                                    },
                                    "english_possessive_stemmer": {
                                      "type":       "stemmer",
                                      "language":   "possessive_english"
                                    },
                                    # Should have synonym filter here
                                    "my_synonym":{
                                            "type": "synonym",
                                            "synonyms_path": "analysis/all_synonym.txt"
                                    },
                                    "my_graph_synonym":{
                                            "type": "synonym_graph",
                                            "synonyms_path": "analysis/all_synonym.txt"
                                    },
                                    "edge_ngram_filter":{
                                        "type": "edge_ngram",
                                        "min_gram": 1,
                                        "max_gram": 10
                                    }
                                  }
                    }
            },
            "mappings":{
                    "properties":{
                            "scene":{
                                    "type": "text",
                                    "analyzer": "analyzer_tfidf",
                                    "search_analyzer": "analyzer_search"
                            },
                            "object_tfidf":{
                                    "type": "text",
                                    "analyzer": "analyzer_tfidf",
                                    "search_analyzer": "analyzer_search"
                            },
                            "object_yolo_term":{
                                    "type": "text",
                                    "analyzer": "analyzer_object_yolo_term",
                                    "search_analyzer": "analyzer_object_yolo_term"
                            },
                            "object_yolo_term_clip":{
                                    "type": "text",
                                    "analyzer": "analyzer_object_yolo_term_clip",
                                    "search_analyzer": "analyzer_search"
                            },
                            "description":{
                                    "type": "text",
                                    "analyzer": "analyzer_object_yolo_term",
                                    "search_analyzer": "analyzer_object_yolo_term"
                            },
                            "description_clip":{
                                    "type": "text",
                                    "analyzer": "analyzer_object_yolo_term_clip",
                                    "search_analyzer": "analyzer_object_yolo_term_clip_search"
                            },
                            "description_clip_tfidf":{
                                    "type": "text",
                                    "analyzer": "analyzer_tfidf",
                                    "search_analyzer": "analyzer_search"
                            },
                            "weekday":{
                                    "type": "text"
                            }
                    }
            }                        
    }#,
    # Will ignore 400 errors, remove to ensure you"re prompted
    #ignore=400
)

####### Add data to es ########
File = "description_all.json"
with open(File) as json_file:
    description = json.load(json_file)

print("Uploading data to the server ...")    

numb_of_image_scan = 0
for id_image_json, content_image_json in tqdm(description.items()):
    numb_of_image_scan += 1
    
    id_image = id_image_json
    
    image_date = id_image[:8] # Extract date information
    image_datetime = datetime.strptime(image_date, '%Y%m%d') # Convert to datetime type
    image_weekday = calendar.day_name[image_datetime.weekday()] # Monday, Tuesday, ...

    scene_image = content_image_json["scene_image"]
    object_image = content_image_json["object_image"]
    object_tfidf_image = content_image_json["object_format_image"]

    document = {
            "id": id_image,
            "scene": scene_image,
            "description": scene_image + ", " + object_image,
            "description_clip": scene_image + ", " + object_image,
            "weekday": image_weekday
    }
    
    # Store document in Elasticsearch
    res = es.index(index=interest_index, doc_type="_doc", id=numb_of_image_scan, body=document)


########### Summary ##############
print("==========\nTotal number of document:")
res= es.search(index=interest_index,body={"query":{"match_all":{}}}, size = 9999) # Simple list all document
print(len(res["hits"]["hits"])) # Number of result


########### Simple Query ##############
# # Directory to images
# Images_Path = "/Volumes/GoogleDrive/My Drive/LSC-test/LSC_DATA/"
#
# input_query = "2 people, cafe, glass, dish"
#
# start_time = time.time()
# query_request_txt, query_request_json = mylib.generate_es_query_dismax_querystringquery(q = input_query,
#                                                                                         list_synonym = List_synonym,
#                                                                                         max_change = 1,
#                                                                                         tie_breaker = 0.7)
# request_result ,id_result = mylib.search_es(es, index = "lsc2019", request = query_request_json,
#                                       percent_thres = 0.5, max_len = 20)
# end_time = time.time()
# print("Search Time: " + str(end_time - start_time) + " seconds.")
#
#
# # Show images
# id_image_path = mylib.add_folder_to_id_images(id_result)
# mylib.show_result(image_path = Images_Path, id_image = id_image_path)



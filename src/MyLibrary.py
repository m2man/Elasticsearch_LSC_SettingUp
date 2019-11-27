#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:58:16 2019
Library for LSC
@author: duynguyen
"""

#from elasticsearch import Elasticsearch
import json
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
from io import BytesIO
import base64
from PIL import Image

#from tqdm import tqdm
#import time

def convert_pil_to_byte64_0(image):
    # Convert PIL to url string that can show in html and store in database
    if image.size[0] > 200:
        image = image.resize((171, 128), Image.ANTIALIAS)
    output = BytesIO()
    image.save(output, format='JPEG')
    image_data = output.getvalue()
    image_data = base64.b64encode(image_data)
    if not isinstance(image_data, str):
        # Python 3, decode from bytes to string
        image_data = image_data.decode()
    data_url = 'data:image/jpg;base64,' + image_data
    return data_url
    # show image in html: <img src = data_url />

def convert_pil_to_byte64(image):
    # Convert PIL to url string that can show in html and store in database
    if image.size[0] > 200:
        image = image.resize((128, 96), Image.ANTIALIAS)
    output = BytesIO()
    image.save(output, format='JPEG')
    image_data = output.getvalue()
    image_64_encode = base64.encodebytes(image_data)
    image_64_encode = image_64_encode.decode("utf-8") # convert into string
    image_64_encode = image_64_encode.replace("\n","") # remove last \n since elastic told that
    return image_64_encode

def unlist(l):
    '''
    Unlist list of lists l into 1 list
    l = [[1, 2], [3]] --> result = [1, 2, 3]
    '''
    result = []
    for s in l:
        result.extend(s)
    return result

def search_es(ES, index, request, percent_thres = 0.9, max_len = 10):
    '''
    ES: Elasticsearch engine with collected data
    index: name of the index in ES
    reques: request in json format
    percent_thers: percentage of maxscore to be a threshold to filt the result
    max_len: maximum result that you want to get
    Output 
    --> res: original result from elasticsearch
    --> result: list of id_image
    '''
    result = None
    
    res= ES.search(index=index, body=request, size = 9999) # Show all result (9999 results if possible)
    numb_result = len(res["hits"]["hits"])
    print("Total searched result: " + str(numb_result))
    
    if numb_result != 0:        
        score = [d["_score"] for d in res["hits"]["hits"]] # Score of all result (higher mean better)
        id_result = [d["_source"]["id"] for d in res["hits"]["hits"]] # List all id images in the result
        
        score_np = np.asarray(score)
        
        max_score = score_np[0]
        thres_score = max_score * percent_thres
        
        index_filter = np.where(score_np > thres_score)[0]
        
        if len(index_filter) > 1:
            result = list(itemgetter(*list(index_filter))(id_result)) 
            numb_result = len(result)
        else:
            result = id_result[0]
            numb_result = 1
        
        if numb_result > max_len:
            result = result[0:max_len]
            
        print("Total remaining result: " + str(numb_result)) # Number of result
        print("Total output result: " + str(min(max_len, numb_result))) # Number of result
    
    return res, result

def show_result(image_path, id_image, ncol = 5):
    if type(id_image) is list:
        images = []
        for i, one_image in enumerate(id_image):
            filename = image_path + one_image
            try:
                images.append(mpimg.imread(filename))
            except FileNotFoundError:
                print("Cannot find ranked: " + str(i) + " - " + one_image)
       
        fig = plt.figure(figsize=(15,15))
        for i, image in enumerate(images):
            ax = fig.add_subplot(len(images) / ncol + 3, ncol, i + 1)
            plt.imshow(image)
            plt.axis('off')
            ax.set_aspect("equal")
    
        plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
    
    if type(id_image) is str:
        filename = image_path + id_image
        try:
            img=mpimg.imread(filename)
            plt.figure()
            plt.imshow(img)
            plt.show()
            plt.axis('off')
        except FileNotFoundError:
            print("Cannot find " + id_image)

def add_folder_to_id_images(id_image):
    # Add link folder to the id_image (based on the first 10 characters)
    
    if type(id_image) is list:
        result = [x[:4] + "-" + x[4:6] + "-" + x[6:8] + "/" + x for x in id_image]        
    else:
        result = id_image[:4] + "-" + id_image[4:6] + "-" + id_image[6:8] + "/" + id_image
    
    return result

def generate_near_query(q, max_change = 1):
    '''
    This function generate some similar query with the query q by changing a little bit in q
    Ex: q = "1 car, 2 person" --> split and generate to "1 car", "2 car", "2 person", "1 person", "3 person"
    max_change >= 0 --> maximum change value for number of object (below and above the original value)
    If max_change = 1, and query is 3 car --> result will be ["2 car", "3 car", "4 car"]
    '''
    result = []
    q_process = q.replace(" and ", ", ") # replace " and " to ", " --> "1 car and 1 person" to "1 car, 1 person"
    q_process = q_process.replace(" or ", ", ") # same with "or" --> we dont care difference between "and" and "or"
    q_process = q_process.replace(", ", ",") # "1 car, 1 person" to "1 car,1 person"        
    q_split = q_process.split(",") # "1 car" "1 person"
    
    if max_change > 0:
        for tag in q_split:
            element = tag.split(" ") # "1 car" --> "1", "car"
            
            try:
                element_numb = int(element[0])
                element_change = [str(element_numb)]
                element_object = ''.join(element[1:])
            except ValueError:
                element_numb = 0
                element_change = []
                element_object = ''.join(element)
            
            # Change original value
            for i in range(1, max_change + 1):
                if (element_numb - i) > 0:
                    element_change.append(str(element_numb - i))
                element_change.append(str(element_numb + i))
            
            result += [x + " " + element_object for x in element_change]
                
    else: # If max_change = 0 --> do not change anything, just split query
        result = q_split
        
    return result    

def generate_es_query_dismax(q, max_change = 1, tie_breaker = 0.7):
    '''
    Generate elastic-formatted request and use the result for the input of elasticsearch
    max_change >= 0: See generate_near_query
    tie_breaker: See elasticsearch document
    Output:
        + request_string is the txt format of the elasticsearch formatted request
        + request_json is the dictionary (json) formatted request that can to be the input for elasticsearch_python that we are using here 
    '''
    
    result = "{\"query\":{\"dis_max\":{\"queries\":["
    result += "{\"multi_match\":{"
    result += "\"query\":" + "\"" + q + "\","
    result += "\"fields\":" + "\"object_yolo_term^3\"," # Here we name the field in the dataset is object_term --> can be change
    result += "\"operator\": \"and\"}}"
    
    result += ",{\"multi_match\":{"
    result += "\"query\":" + "\"" + q + "\","
    result += "\"fields\":" + "\"object_yolo_term_clip^2\","
    result += "\"operator\": \"and\"}}"
    
    query_similar = generate_near_query(q, max_change = max_change)
    
    for sub_query in query_similar:
        result += ",{\"match\":{"
        result += "\"object_yolo_term\":" + "\"" + sub_query + "\"}}"
        
        result += ",{\"match\":{"
        result += "\"object_yolo_term_clip\":" + "\"" + sub_query + "\"}}"
    
    result += "],\"tie_breaker\":" + str(tie_breaker)
    result += "}}}"
    
    request_string = result
    request_json = json.loads(request_string)
    
    return request_string, request_json

def generate_original_synonym(list_synonym, given_word):
    '''
    Generate the synonym of given_word and these synonym are the name of the classes in yolo/cbnet
    list_synonym: is the list of synonym generated by Glove.iypn from colab, then download
    Output:
        + result: list of original synonym word of given_word
        + If can find any synonym --> len = 0
    '''
    result = []
    for index, sublist in enumerate(list_synonym):
        if given_word in sublist:
            result.append(list_synonym[index][0])
    return result

def generate_subterm_query(list_synonym, q):
    '''
    divided query q into subterm then find original synonym term for each subterm, and return the result
    list_synonym: see generate_original_synonym
    For ex: q = "1 car, 2 person" --> result = [[1 car, 1 truck, 1 motorbike], [2 person, 2 people]]
    Also give the original term --> original_result = [1 car, 2 person] # No synonym
    '''
    
    result = []
    q_process = q.replace(" and ", ", ") # replace " and " to ", " --> "1 car and 1 person" to "1 car, 1 person"
    q_process = q_process.replace(" or ", ", ") # same with "or" --> we dont care difference between "and" and "or"
    q_process = q_process.replace(", ", ",") # "1 car, 1 person" to "1 car,1 person"        
    q_split = q_process.split(",") # "1 car" "1 person"
    original_result = [[x] for x in q_split]
    
    for tag in q_split:
        element = tag.split(" ") # "1 car" --> "1", "car" 
        
        try: # Split number and object
            element_numb = int(element[0])
#            element_change = [str(element_numb)]
#            element_object = ''.join(element[1:])
            element_object = " ".join(element[1:])
        except ValueError:
            element_numb = 0
#            element_change = ""
#            element_object = "".join(element)
            element_object = " ".join(element)  
        
        original_synonym = generate_original_synonym(list_synonym, element_object) # Find the original synonym
        if len(original_synonym) == 0:
            original_synonym = [element_object]
        
        if element_numb == 0:
            result.append([x for x in original_synonym])
        else:
            result.append([str(element_numb) + " " + x for x in original_synonym])
            
    return original_result, result

def generate_querystringquery_and_subquery(sq, max_change = 1):
    '''
    Generate query string format from the subterm query sq (generated from generate_subterm_query)
    Also generate subquery from the subterm query sq (quite the same with generate_near_query)
    max_change >= 0 --> maximum change value for number of object (below and above the original value)
    For Ex: sq = [[1 car, 1 truck, 1 motorbike], [2 person]], max_change = 1
    query_string_full = "("1 car" OR "1 truck" OR "1 motorbike") AND ("2 person" OR "2 people")"
    query_string_term = [[("1 car" OR "1 truck" OR "1 motorbike"), ("2 person" OR "2 people")]] --> subterm of query_string_full
    query_string_term_adjust = [[("2 car" OR "2 truck" OR "2 motorbike")], [ ... ]] --> adjust quantity with max_change
    '''
    
    query_string_full = ""
    query_string_term_adjust = []
    query_string_term = []
    
    change_seq = np.linspace(-max_change, max_change, 2 * max_change + 1)
    change_seq = np.delete(change_seq, max_change) # remove 0 out of vector --> we dont keep original term
    
    for term in sq:
        query_string = "("
        
        for subterm in term:
            query_string += "(" + subterm + ")" + " OR "
            
#            # Generate sub queries
#            element = subterm.split()
#            try:
#                element_numb = int(element[0])
#                element_object = ''.join(element[1:])
#                element_change = element_numb + change_seq
#                element_change = np.delete(element_change, np.where(element_change <= 0)[0]) # remove negative and 0 number
#                element_change = list(element_change)
#                subquery += [str(int(x)) + " " + element_object for x in element_change]
#            except ValueError:
#                element_numb = 0
#                element_object = ''.join(element)
#                element_change = []
#                subquery += [element_object]
            
        query_string = query_string[0:-4] + ")"
        query_string_term.append([query_string])
        query_string_full += query_string + " AND "
    
    for term in query_string_term:
        assert(len(term) == 1)
        term_str = term[0]
        term_str = term_str.replace("(", "")
        quantity = term_str.split()
        try:
            quantity = int(quantity[0])
            quantity_change = quantity + change_seq
            quantity_change = np.delete(quantity_change, np.where(quantity_change <= 0)[0]) # remove negative and 0 number
        except ValueError:
            quantity_change = np.array([])
        if (len(quantity_change) > 0):
            for i in quantity_change:
                term_change = term[0].replace(str(quantity), str(int(i)))
                query_string_term_adjust.append([term_change])
        
    query_string_full = query_string_full[0:-5]
    
    return query_string_full, query_string_term, query_string_term_adjust

def generate_es_query_dismax_querystringquery(q, list_synonym, max_change = 1, tie_breaker = 0.7):
    '''
    Quite the same with generate_es_query_dismax but now inclide query string query, not multimatch anymore
    Generate elastic-formatted request and use the result for the input of elasticsearch
    list_synonym: list of synonym generated from Glove and only support classes in yolo or cbnet or your own defined
    max_change >= 0: See generate_near_query
    tie_breaker: See elasticsearch document
    Output:
        + request_string is the txt format of the elasticsearch formatted request
        + request_json is the dictionary (json) formatted request that can to be the input for elasticsearch_python that we are using here 
    '''
    
    # Step 1: generate subterm and unlist into original subterm
    o_subterm, subterm = generate_subterm_query(list_synonym, q)
    
    # Step 2: generate query string query
    qsq, qs_term, qs_term_adjust = generate_querystringquery_and_subquery(subterm, max_change)
    o_qsq, o_qs_term, o_qs_term_adjust = generate_querystringquery_and_subquery(o_subterm, max_change)
    
    # Step 3: concacate to json or txt file
    # Step 3.1: define main part
    result = "{\"query\":{\"dis_max\":{\"queries\":["
    
    # Step 3.2: 1st query: constrain query string query (a & b & c) --> lower case --> without synonym term (only original term)
    # Here we search in description term --> exactly the same search term (without synonym) will get score
    result += "{\"query_string\":{"
    result += "\"query\":" + "\"" + o_qsq + "\","
    result += "\"default_field\":" + "\"description\"," # Here we name the field in the dataset is ... --> can be change
    result += "\"boost\":" + "3}}"

    result += ",{\"query_string\":{"
    result += "\"query\":" + "\"" + qsq + "\","
    result += "\"default_field\":" + "\"gps_description\","  # Here we name the field in the dataset is ... --> can be change
    result += "\"boost\":" + "24}}"

    # Step 3.3: 2nd query: constrain query string query (A & B & C) --> upper case --> synonym term
    result += ",{\"query_string\":{"
    result += "\"query\":" + "\"" + qsq + "\","
    result += "\"default_field\":" + "\"description\"," # Here we name the field in the dataset is ... --> can be change
    result += "\"boost\":" + "2}}" 
    
    # Step 3.4: 3rd query: constrain query string query (a & b & c) --> lower case --> without synonym term (only original term)
    # Here we search in description_clip --> can be noise by other quantity of other objects
    result += ",{\"query_string\":{"
    result += "\"query\":" + "\"" + o_qsq + "\","
    result += "\"default_field\":" + "\"description_clip\"," # Here we name the field in the dataset is ... --> can be change
    result += "\"boost\":" + "1.25}}"
    
    # Step 3.5: 2nd --> query: sub query string term with boost = 2 (each query for A, B, C)
    for sub_query, o_sub_query in zip(qs_term, o_qs_term):
        result += ",{\"query_string\":{"
        result += "\"query\":" + "\"" + sub_query[0] + "\","
        result += "\"default_field\":" + "\"description\"," # Here we name the field in the dataset is ... --> can be change
        result += "\"boost\":" + "0.75}}"
        
        result += ",{\"query_string\":{"
        result += "\"query\":" + "\"" + o_sub_query[0] + "\","
        result += "\"default_field\":" + "\"description_clip\"," # Here we name the field in the dataset is ... --> can be change
        result += "\"boost\":" + "0.45}}"
    
    for sub_query, o_sub_query in zip(qs_term_adjust, o_qs_term_adjust):
        result += ",{\"query_string\":{"
        result += "\"query\":" + "\"" + sub_query[0] + "\","
        result += "\"default_field\":" + "\"description\"," # Here we name the field in the dataset is ... --> can be change
        result += "\"boost\":" + "0.35}}"
        
        result += ",{\"query_string\":{"
        result += "\"query\":" + "\"" + o_sub_query[0] + "\","
        result += "\"default_field\":" + "\"description_clip\"," # Here we name the field in the dataset is ... --> can be change
        result += "\"boost\":" + "0.1}}"
        
#        result += ",{\"match\":{"
#        result += "\"object_yolo_term_clip\":" + "\"" + sub_query + "\"}}"
        
    # Step 4: close the query with other parameters
    result += "],\"tie_breaker\":" + str(tie_breaker)
    result += "}}}"
    
    request_string = result
    request_json = json.loads(request_string)
    
    return request_string, request_json

def find_descriptive_attribute_in_list_images(database, list_images):
    '''
    Find all descriptive attributes (or something relavant) of list_images in the database (description json file storing all information of all images)
    Descriptive value is defined as below output
    Output:
        + result_high is a list of most appearance (except the one that all images contain) attribute/environment
        + result_low is a list of least common (most distinctive value) and dont take attribute appearing once
    '''
    
#    # Only for debug
#    from random import seed
#    from random import randint
#    
#    seed(3)
#    random_index = [randint(0, len(database)-1) for i in range(15)]
#    all_id_images = np.array(list(database.keys()))
#    list_images = all_id_images[random_index]
#    # End debug
    
    scene_list_images = [database[x]['scene_image'].split(', ') for x in list_images]
    
    list_scene = []
    for l in scene_list_images:
        list_scene.extend(l)
        
    scene_numb = Counter(list_scene) 
    len_list_images = len(list_images)
    
    scene_appearance = list(scene_numb.keys())
    numb_appearance = np.array(list(scene_numb.values())) # array of numb of appearance of each scence
    threshold_high = np.quantile(numb_appearance, 0.8) # only keep 20% of most common
    threshold_low = np.quantile(numb_appearance, 0.2) # only keep 20% least common --> likely that it will be one
    
    if threshold_high == len_list_images: # if threshold = len(list_images) --> all images have that att --> useless! --> remove that value
        threshold_high = np.max(numb_appearance[np.where(numb_appearance < len_list_images)])
        
    if threshold_low == 1: # if thershold = 1 --> only 1 picture contains that value --> seem to be useless and not disticntive --> find the 2nd smallest
        threshold_low = np.min(numb_appearance[np.where(numb_appearance > 1)])
        
    select_index_high = list(np.where((numb_appearance >= threshold_high) & (numb_appearance < len_list_images))[0])
    select_index_low = list(np.where((numb_appearance <= threshold_low) & (numb_appearance > 1))[0])
    
    result_high = []
    for x in select_index_high:
        result_high.append(scene_appearance[x])
        
    result_low = []
    for x in select_index_low:
        result_low.append(scene_appearance[x])
    
    return result_low, result_high

def search_es_embedding(ES, index, embedded_query, percent_thres = 0.9, max_len = 10):

    '''
    Same with search_es but this is used for embedding --> request part will be different
    embedded_query is a vector(as tolist() form)
    '''

    result = None

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_embedded, doc['description_embedded']) + 1.0",
                "params": {"query_embedded": embedded_query}
            }
        }
    }

    res = ES.search(index=index,
                    body={
                        "query": script_query,
                        "_source": {"includes": ["id", "description"]}
                    },
                    size=9999)  # Show all result (9999 results if possible)

    numb_result = len(res["hits"]["hits"])
    print("Total searched result: " + str(numb_result))

    if numb_result != 0:
        score = [d["_score"] for d in res["hits"]["hits"]]  # Score of all result (higher mean better)
        id_result = [d["_source"]["id"] for d in res["hits"]["hits"]]  # List all id images in the result

        score_np = np.asarray(score)

        max_score = score_np[0]
        thres_score = max_score * percent_thres

        index_filter = np.where(score_np > thres_score)[0]

        if len(index_filter) > 1:
            result = list(itemgetter(*list(index_filter))(id_result))
            numb_result = len(result)
        else:
            result = id_result[0]
            numb_result = 1

        if numb_result > max_len:
            result = result[0:max_len]

        print("Total remaining result: " + str(numb_result))  # Number of result
        print("Total output result: " + str(min(max_len, numb_result)))  # Number of result

    return res, result
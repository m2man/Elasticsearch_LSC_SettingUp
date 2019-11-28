import json
import pickle
from elasticsearch import Elasticsearch
from tqdm import tqdm
from datetime import datetime
import calendar
Folder_Path = '/Users/duynguyen/DuyNguyen/GitKraken/Elasticsearch_LSC_SettingUp/'

File = Folder_Path + 'data/bow_my_dictionary.pickle'
with open(File, 'rb') as f:
    my_dictionary = pickle.load(f)
numb_bow_ft = len(my_dictionary)

File = Folder_Path + 'data/combined_description_all.json'
with open(File) as json_file:
    combined_description = json.load(json_file)

File = Folder_Path + 'data/bow_feature_all.pickle'
with open(File, 'rb') as f:
    bow_ft_all = pickle.load(f)

###### Connect to the elastic cluster --> run elasticsearch first #########
es = Elasticsearch([{"host": "localhost", "port": 9200}])

interest_index = "lsc2019_bow"

print("Deleting index: " + interest_index)
try:
    es.indices.delete(index=interest_index)
except Exception:
    print("Do not have index to delete: " + interest_index)

print("Creating index: " + interest_index)
es.indices.create(
    index=interest_index,
    body={
        "settings": {
            # just one shard, no replicas for testing
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings":{
            "properties":{
                "description":{
                    "type": "text",
                },
                "weekday":{
                    "type": "text"
                },
                "description_embedded":{
                    "type": "dense_vector",
                    "dims": numb_bow_ft
                }
            }
        }
    }
)

print("Uploading data to the server ...")

numb_of_image_scan = 0
for id_image_json, content_image_json in tqdm(combined_description.items()):
    numb_of_image_scan += 1

    id_image = id_image_json

    image_date = id_image[:8]  # Extract date information
    image_datetime = datetime.strptime(image_date, '%Y%m%d')  # Convert to datetime type
    image_weekday = calendar.day_name[image_datetime.weekday()]  # Monday, Tuesday, ...

    description_image = content_image_json

    document = {
        "id": id_image,
        "description": description_image,
        "weekday": image_weekday,
        "description_embedded": bow_ft_all[id_image].tolist()
    }

    # Store document in Elasticsearch
    res = es.index(index=interest_index, doc_type="_doc", id=numb_of_image_scan, body=document)

########### Summary ##############
print("==========\nTotal number of document:")
res= es.search(index=interest_index,body={"query":{"match_all":{}}}, size = 9999) # Simple list all document
print(len(res["hits"]["hits"])) # Number of result
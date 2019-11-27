'''
This is for testing ES for searching with embedding text ranking by cosine similarity
More info: https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch
'''

import json
import pickle
import random
from tqdm import tqdm
from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])

interest_index = "lsc2019_embedding"


es.indices.delete(index=interest_index)
es.indices.create(
    index=interest_index,
    body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "dynamic": "true",
            "_source": {
                "enabled": "true"
            },
            "properties": {
                "description":{
                    "type": "text",
                },
                "description_embedded": {
                    "type": "dense_vector",
                    "dims": 768
                }
            }
        }
    }
)

print("Loading data file")
File = "description_all.json"
with open(File) as json_file:
    description = json.load(json_file)

File = "BERT_Embedded_Annotation.pickle"
with open(File, "rb") as pickle_file:
    description_embedded = pickle.load(pickle_file)
print("Finished loading data file")

# Get id images, both json and embedded file should have the same id list
list_id_images = list(description.keys())

number_of_files_want_to_index = len(list_id_images)
#list_id_images_index_shuffle = [x for x in range(len(list_id_images))]
#random.shuffle(list_id_images_index_shuffle)
#index_random = list_id_images_index_shuffle[:number_of_files_want_to_index]
#list_id_images_random = [list_id_images[x] for x in index_random]
list_id_images_random = list_id_images

for number_of_images_scanned in tqdm(range(number_of_files_want_to_index)):
    id_image = list_id_images_random[number_of_images_scanned]
    scene_image = description[id_image]["scene_image"]
    object_image = description[id_image]["object_image"]

    document = {
        "id": id_image,
        "description": scene_image + ", " + object_image,
        "description_embedded": description_embedded[id_image].tolist()
    }

    res = es.index(index=interest_index,
                   doc_type="_doc",
                   id=number_of_images_scanned,
                   body=document)



File = "BERT_Model.pickle"
with open(File, "rb") as pickle_file:
    bert_model = pickle.load(pickle_file)
print("Finished loading embedding model")

query = 'automobile'
query_embedded = bert_model.encode(query)[0].tolist()

script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_embedded, doc['description_embedded']) + 1.0",
            "params": {"query_embedded": query_embedded}
        }
    }
}

response = es.search(
    index=interest_index,
    body={
        "query": script_query,
        "_source": {"includes": ["id", "description"]}
    },
    size=10
)

print("{} total hits in {} ms.".format(response["hits"]["total"]["value"], response['took']))
for hit in response["hits"]["hits"]:
    print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
    print(hit["_source"])
    print()
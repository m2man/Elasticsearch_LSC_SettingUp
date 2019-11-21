# Elasticsearch_LSC_SettingUp
This is for setting up ES. Please install Elasticsearch first. Then run the ES, it will create a localhost:9200. Finally run this script to create analysers and tokeniser for fields in the ES.

### Indexing
Run the **ES_text_search.py** file to start indexing data. All information of LSC data is store in the **description_all.json** file. Please make sure that you have already run the elasticsearch before running this code. You also want to install ***Kibana*** to have more config on the Elasticsearch engine. 

After running the code, please wait for processing and indexing data to finish. Then you can play with the Elasticsearch

### Run with Django
If you want to run it with Django, in the **views.py**, use this code:

```
import MyLib as mylib
# query is the string query
# List_synonym is the synonym file provided in the repo
# output will be result including 2 information for each image (image_path and name of that image)
data, _ = mylib.generate_es_query_dismax_querystringquery(q=query,
                                                                  list_synonym=List_synonym,
                                                          max_change=1,
                                                          tie_breaker=0.7,
                                                          numb_get_result=100)
headers = {"Content-Type": "application/json"}
response = requests.post("http://localhost:9200/lsc2019/_search", headers=headers, data=data)

if response.status_code == 200:
	#stt = "Success"
	response_json = response.json()  # Convert to json as dict formatted
	id_images = [d["_source"]["id"] for d in response_json["hits"]["hits"]]
	id_images_full_path = mylib.add_folder_to_id_images('./LSC_Data/', id_images)
	result = [{'img': id_images_full_path[x], 'title': id_images[x]} for x in range(len(id_images))]
	#print("Searching time: " + str(response_json["took"]) + "ms")
else:
	#stt = "Fail"
	result = []

```

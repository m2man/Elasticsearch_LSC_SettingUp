# Elasticsearch_LSC_SettingUp
This is for setting up ES. Please install Elasticsearch first. Then run the ES, it will create a localhost:9200. Finally run this script to create analysers and tokeniser for fields in the ES.
***Note: Please copy the [all_synonym.txt] file to [elastic_folder/config/analysis/] folder before running***

### Indexing
**28/11/2019 Update**
1. Now add 3 ES setting up files: text, bow, combined. Run the **ES_[method]_search.py** to begin.
2. Change to MyLibrary_v2 (change name of some function as well)
3. Run the Process_BOW to create all necessary files (bow feature, bow dictionary, ...)
After running the code, please wait for processing and indexing data to finish. Then you can play with the Elasticsearch

### Run with Django
If you want to run it with Django, in the **views.py**, use this code:

```
import MyLibrary_v2 as mylib_v2
# query is the string query
# List_synonym is the synonym file provided in the repo
# output will be result including 2 information for each image (image_path and name of that image)
data  = mylib_v2.generate_es_text(q=query)
headers = {"Content-Type": "application/json"}
response = requests.post("http://localhost:9200/lsc2019/_search", headers=headers, data=data)

if response.status_code == 200:
	#stt = "Success"
	response_json = response.json()  # Convert to json as dict formatted
	id_images = [d["_source"]["id"] for d in response_json["hits"]["hits"]]
	id_images_full_path = mylib_v2.add_folder_to_id_images('./LSC_Data/', id_images)
	result = [{'img': id_images_full_path[x], 'title': id_images[x]} for x in range(len(id_images))]
	#print("Searching time: " + str(response_json["took"]) + "ms")
else:
	#stt = "Fail"
	result = []

```

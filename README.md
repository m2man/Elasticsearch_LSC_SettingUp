# Elasticsearch_LSC_SettingUp
This is for setting up ES. Please install Elasticsearch first. Then run the ES, it will create a localhost:9200. Finally run this script to create analysers and tokeniser for fields in the ES.
***Note: Please copy the [all_synonym.txt] file to [elastic_folder/config/analysis/] folder before running***

## Change Log
### 09/03/2020
1. Merge all retrieving function to **MyLibrary_v2.py**
2. Grouping now is performed at indexing procedure (No need to run grouping after searching) --> Update **ES_text_search.py**
3. Add searching through REST API to make it faster (reduce searching time from 1.5s to below 0.4s)
4. Try **Test_sequence_query.py** to measure the time for sequential event retrieving

### 25/02/2020
#### How to search Sequence Action
1. Need to re-index elastic database (add some **time** fields)
2. Download the SIFT feature pickle (below) and put in **data** folder
3. Try with **Test_sequence_query.py**
4. The search flow as depicted in below figures.
<img width="673" alt="Screenshot 2020-02-28 at 15 28 44" src="https://user-images.githubusercontent.com/15571804/75561615-382af400-5a3f-11ea-9a3f-f27860e87233.png">

<img width="782" alt="Screenshot 2020-02-28 at 15 28 54" src="https://user-images.githubusercontent.com/15571804/75561694-5abd0d00-5a3f-11ea-83a9-ce26768cb6d1.png">


#### Updated Note
1. Add NLP Processing in **Tag_Event.py** Library.
- Usage: Run the function ```extract_info_from_sentence_full_tag(sentence)``` to extract the dictionary of past, present, and future action. If there is no information extracted, it will be an empty dict for each action.
2. Add **ProcImgLib.py** Library: grouping list of images based on its SIFT, description, and time into distinct clusters.
- Usage: Run the function ```grouping_image_with_sift_dict(list_images, description_dict, sift_dict)``` to group images. The result will be the list of images cluster.
- ```Description_dict``` can be found in folder **data** and ```Sift_dict```[Download](https://drive.google.com/file/d/1QI79-WYDkgW7GjCQ7Kl49YjNeUxK7B0m/view?usp=sharing).
3. Update **MyLibrary_v2.py**: Mostly update about generate query in search text case
- Mechanism will be generate list of dismax and filter list, then generate json format query to pass to ES
- ```generate_list_dismax_part_and_filter_time_from_info(info)``` take the input is the past/present/future action created in section 1, and create list dismax, filter.
- Pass the list dismax and filter to ```generate_query_text``` function to create json fornat to be used in ES
4. **Test_sequence_query.py** to test to search sequence (or single) action in ES. Flow will be generate dictionary of past/present/future dict, then seach each action and update the following. Do it TWICE!
- **Single action**: search text as normal
- **Double action**: search the previous action --> group result images --> generate times --> Add times as condition to the following action --> Do it again 1 more time!
	- The Result will be the ranked list of [[group action 1], [group action 2], average score]
- **Triple action**: Search action 1, action 3 --> group 1, group 3 --> times 1, times 3 --> add to query 2 --> search action 2
5. **ES_text_search.py** update how to index data: add time, day, month, hour, minute field to the data.

### 05/12/2019
1. Fix bug in text search (in synonym dictionary)
2. Update **all_synonym.txt** (add it to config folder in Elasticsearch)
3. Update **List_synonym_glove_all.pickle**
4. Revise setting in **ES_text_search.py** and **ES_combined_search.py** (add ```expand=false``` setting in synonym filter)

### 28/11/2019
1. Now add 3 ES setting up files: text, bow, combined. Run the **ES_[method]_search.py** to begin.
2. Change to MyLibrary_v2 (change name of some function as well)
3. Run the Process_BOW to create all necessary files (bow feature, bow dictionary, ...)
After running the code, please wait for processing and indexing data to finish. Then you can play with the Elasticsearch

## Run with Django
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

'''
Read the scene and attribute detection json file and create new one by making a sentence for each image
Remove some attribute that all images contain --> useless
'''

import json
import numpy as np
from collections import Counter

File = '/Volumes/GoogleDrive/My Drive/LSC-test/att.json'

with open(File) as json_file:
    data = json.load(json_file)

# Check the format of idimage (data.keys()) --> since some of them include the folder name
id_images = list(data.keys())
len_each_id = [len(x) for x in id_images]
set(len_each_id) #  --> should be 1 number "23" (here is 34) --> take id[11:]

new_data = {}

print("Processing ...")
for image_name, content in data.items():
    id_image = image_name
    if len(id_image) > 23:
        id_image_short = id_image[(len(id_image) - 23):]
    else:
        id_image_short = id_image
        
#    print('Process: ' + id_image_short)
    
    content_str = ''
    if len(data.get(id_image, "none")) > 0:
        for att, quant in content.items():
            att = att.replace("-", ", ")
            att = att.replace("/", ", ")
            att = att.replace("_", ", ")
            # Find unique attribute since some replicate
            att_split = att.split(", ")
            unique_att = list(set(att_split))
            content_str = content_str + ", ".join(unique_att) + ', '
        content_str = content_str[:-2] # Remove the ', ' at the end of the sentence
        content_split = content_str.split(", ")
        unique_content = list(set(content_split))
        content_str = ", ".join(unique_content)
        #content_str = content_str + '.' # Add dot at the end of the sentence
    content_image = content_str

    new_data[id_image_short] = content_image

# Post process --> remove attibutes that all images contain
print("Post Processing ...")
att_all = [new_data[x].split(", ") for x in list(new_data.keys())]
numb_images = len(att_all)
list_att_all = []
for i in att_all:
    list_att_all.extend(i)
att_counter = Counter(list_att_all)

att_appearance = list(att_counter.keys())
numb_appearance = np.array(list(att_counter.values()))

index_remove_att = list(np.where(numb_appearance == numb_images)[0])
list_remove_att = [att_appearance[x] for x in index_remove_att]

result = {}
for image_name, content in new_data.items():
    id_image = image_name
    content_split = content.split(", ")
    content_after_remove = [x for x in content_split if x not in list_remove_att]
    content_str = ", ".join(content_after_remove)
    result[id_image] = content_str
    
    
with open('attribute_all.json', 'w') as outfile:
    json.dump(result, outfile)



#att = json.load(open("attribute_all.json"))
#att_all = [att[x].split(", ") for x in list(att.keys())]
#list_att_all = []
#for i in att_all:
#    list_att_all.extend(i)
#att_counter = Counter(list_att_all)
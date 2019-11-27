'''
Read the object detection json file and create new one by making a sentence for each image
'''

import json 

File = '/Volumes/GoogleDrive/My Drive/LSC-test/cbnet_all.json'

with open(File) as json_file:
    data = json.load(json_file)

# Check the format of idimage (data.keys()) --> since some of them include the folder name
#id_images = list(data.keys())
#len_each_id = [len(x) for x in id_images]
#set(len_each_id) #  --> should be 1 number (here is 23, 34) --> take id[11:]

# keys_list = list(data.keys())

##### Concatenate data #####
# Format: 1 car, 2 person, 2 pen, ...
    
new_data = {}

for image_name, content in data.items():
    id_image = image_name
    if len(id_image) > 23:
        id_image_short = id_image[(len(id_image) - 23):]
    else:
        id_image_short = id_image
        
    print('Process: ' + id_image_short)
    
    if len(data.get(id_image, "none")) == 0:
        content_str = 'Nothing'
    else:
        content_str = ''
        for object, quant in content.items():
            content_str = content_str + str(quant) + ' ' + object + ', '
        content_str = content_str[:-2] # Remove the ', ' at the end of the sentence
        #content_str = content_str + '.' # Add dot at the end of the sentence
    content_image = content_str

    new_data[id_image_short] = content_image

with open('object_cbnet_all.json', 'w') as outfile:
    json.dump(new_data, outfile)

##### New format of concat data ######
# Format: 1 car, 2 person person, 2 pen pen, ...
# Another try: 5 many person person person person person (threshold to appear many can be 5 or 6, ...)
# Mulple object appear --> easy to detect
    
threshold_many = 3 # if number of object is >= threshold_many --> add many to the text

new_data_format = {}

for image_name, content in data.items():
    id_image = image_name
    if len(id_image) > 23:
        id_image_short = id_image[(len(id_image) - 23):]
    else:
        id_image_short = id_image
        
    print('Process: ' + id_image_short)
    
    if len(data.get(id_image, "none")) == 0:
        content_str = 'Nothing'
    else:
        content_str = ''
        for object, quant in content.items():
            content_str = content_str + str(quant) + ' ' + object + ' '
            for i in range(1, quant):
                if quant >= threshold_many and i == (threshold_many - 1):
                    content_str = content_str + 'many '
                content_str = content_str + object + ' '
            content_str = content_str[:-1] # Remove the last ' '
            content_str = content_str + ', ' # Then add ,
        content_str = content_str[:-2] # Remove the ', ' at the end of the sentence
        #content_str = content_str + '.' # Add dot at the end of the sentence
    content_image = content_str

    new_data_format[id_image_short] = content_image

with open('object_cbnet_all_extend.json', 'w') as outfile:
    json.dump(new_data_format, outfile)
import requests
import pickle
server = 'https://nominatim.openstreetmap.org/reverse?format=geojson'

def get_location_from_gps(lat, lon):
    #lat = 44.4
    #lon = 11.3
    query_format = f'&lat={lat}&lon={lon}'
    query_url = server+query_format
    res = requests.post(server+query_format)
    result = res.json()
    display_name = result['features'][0]['properties']['display_name']
    # Lower
    display_name = display_name.lower()
    # Remove "-"
    display_name = display_name.replace('-', ' ')
    # Make it to list
    location_parts = display_name.split(', ')
    # Remove number
    location_no_number = []
    for x in location_parts:
        try:
            k = int(x)
        except:
            location_no_number.append(x)
    return location_no_number

'''
q = 'Hà Nội - Ba Đình - Trúc Bạch'
q = q.replace(' ', '+')
q = q.split('+-+')
q = '+'.join(q)
query_final = f'https://nominatim.openstreetmap.org/search?q={q}&format=geojson'
res = requests.post(query_final)
'''

# location_origin = ['dublin city university (dcu)', 'invent cafe', 'work', 'dcu restaurant', 'dunnes cornelscourt',
#                    'omni park shopping centre', 'clarion hotel & congress oslo airport', 'oslo lufthavn gardermoen, gate 53', 'fjellheisen cable car tromsø', 'captain americas cookhouse & bar', 
#                    'tromsø lufthavn, langnes (tos)', 'verbena avenue', 'rathdown park', 'hotel killarney', 'oslo - gardermoen norway international airport, sas domestic lounge osl', 
#                    'b&q', 'clarion hotel the edge', 'science gallery trinity college dublin', 'jervis shopping centre', 'the westin', 
#                    'rå sushi tromsø', 'grafton street', 'connolly train station', 'tea & tales', 'howth junction business centre', 
#                    'power city', 'ross castle', 'daa executive lounge', 'supervalu, killester', 'sas gold lounge arlanda', 
#                    'sas domestic lounge osl', 'forskningsparken i tromsø', 'sas businesslounge tromsø lufthavn', 'the henry grattan', 'home', 
#                    'dublin airport (dub) - aerfort bhaile átha cliath', 'gate 110', 'the black sheep', 'costa coffee', 'stockholm arlanda (arn) international airport', 
#                    'gate 102', 'the helix', 'oslo - gardermoen norway international airport, clarion hotel & congress oslo airport', 'the blanchardstown centre', 'oslo - gardermoen norway international airport', 
#                    'smarthotel', 'yamamori izakaya bar', 'stockholm-arlanda airport gate 10a', 'realfagbygget', 'radisson blu hotel, tromsoe', 
#                    'pavilions shopping centre', 'donaghmede shopping centre', 'mcdonalds', 'thunder road café', 'arnotts department store', 
#                    'mh-kafeen uit', 'dunnes stores', 'park inn radisson hotel, gardermoen', 'oslo lufthavn (osl)', 'siam thai, dundrum town centre', 'the sisters home']

# useless_char = ["(", ")", ",", " -", " &"]
# semantic_location = {}

# for x in location_origin:
#     remove_specific_char = x
#     for y in useless_char:
#         remove_specific_char = remove_specific_char.replace(y, "")
#     remove_specific_char = remove_specific_char.replace("-", " ")
#     split_char = remove_specific_char.split()
#     semantic_location[x] = split_char

# # Manual
# semantic_location['invent cafe'] += ['coffee shop', 'dcu']
# semantic_location['dunnes cornelscourt'] += ['stores', 'shopping center']
# semantic_location['oslo lufthavn gardermoen, gate 53'] += ['oslo airport', 'norway']
# semantic_location['fjellheisen cable car tromsø'] += ['aerial tramway', 'tromso', 'norway']
# semantic_location['captain americas cookhouse & bar'] += ['american restaurant']
# semantic_location['tromsø lufthavn, langnes (tos)'] += ['tromso airport']
# semantic_location['verbena avenue'] += ['street']
# semantic_location['rathdown park'] += ['street']
# semantic_location['b&q'] += ['warehouse', 'store']
# semantic_location['science gallery trinity college dublin'] += ['tcd', 'university']
# semantic_location['the westin'] += ['hotel']
# semantic_location['rå sushi tromsø'] += ['sushi restaurant', 'tromso', 'norway']
# semantic_location['tea & tales'] += ['coffee shop', 'tea shop', 'sandwich shop', 'bakery shop']
# semantic_location['howth junction business centre'] += ['car park', 'entrance', 'dart station']
# semantic_location['power city'] += ['electronic store', 'electronic shop']
# semantic_location['ross castle'] += ['heritage', 'historic accommodation', 'national park']
# semantic_location['daa executive lounge'] += ['dublin airport']
# semantic_location['supervalu, killester'] += ['supermarket', 'groceries']
# semantic_location['sas gold lounge arlanda'] += ['stockholm arlanda airport', 'sweden']
# semantic_location['sas domestic lounge osl'] += ['oslo airport', 'norway']
# semantic_location['forskningsparken i tromsø'] += ['innovation center', 'norway', 'tromso']
# semantic_location['sas businesslounge tromsø lufthavn'] += ['business lounge', 'tromso airport', 'norway']
# semantic_location['the henry grattan'] += ['restaurant']
# semantic_location['gate 110'] += ['airport']
# semantic_location['the black sheep'] += ['restaurant', 'dublin']
# semantic_location['costa coffee'] += ['coffee shop']
# semantic_location['gate 102'] += ['airport']
# semantic_location['the helix'] += ['venue', 'theater', 'conference center', 'coffee shop']
# semantic_location['the blanchardstown centre'] += ['shopping center']
# semantic_location['smarthotel'] += ['hotel', 'accomodation', 'oslo', 'norway']
# semantic_location['yamamori izakaya bar'] += ['sake bar', 'restaurant']
# semantic_location['stockholm-arlanda airport gate 10a'] += ['stockholm arlanda airport']
# semantic_location['realfagbygget'] += ['natural science building', 'university of bergen']
# semantic_location['radisson blu hotel, tromsoe'] += ['hotel', 'tromso', 'norway']
# semantic_location['mcdonalds'] += ['restaurant']
# semantic_location['thunder road café'] += ['bar', 'burger restaurant']
# semantic_location['arnotts department store'] += ['shopping center']
# semantic_location['mh-kafeen uit'] += ['coffee shop', 'uit university', 'tromso', 'norway']
# semantic_location['dunnes stores'] += ['department store', 'shopping center']
# semantic_location['park inn radisson hotel, gardermoen'] += ['oslo airport', 'norway']
# semantic_location['oslo lufthavn (osl)'] += ['oslo airport', 'norway']
# semantic_location['siam thai, dundrum town centre'] += ['restaurant']
# semantic_location['the sisters home'] += ['another house', 'someone house']

# with open('semantic_location.pickle', 'wb') as f:
#     pickle.dump(semantic_location, f)

with open('semantic_location.pickle', 'rb') as f:
    semantic_location = pickle.load(f)
import requests
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
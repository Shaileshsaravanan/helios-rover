import requests

def get_path_coordinates(api_key, start_location, end_location, waypoints=None):
    url = 'https://maps.googleapis.com/maps/api/directions/json'
    params = {
        'origin': start_location,
        'destination': end_location,
        'key': api_key
    }
    if waypoints:
        params['waypoints'] = '|'.join(waypoints)

    response = requests.get(url, params=params)
    directions = response.json()

    if directions['status'] != 'OK':
        raise Exception(f"Error fetching directions: {directions['status']}")

    path = []
    for leg in directions['routes'][0]['legs']:
        for step in leg['steps']:
            start_lat = step['start_location']['lat']
            start_lng = step['start_location']['lng']
            path.append((start_lat, start_lng))

            end_lat = step['end_location']['lat']
            end_lng = step['end_location']['lng']
            path.append((end_lat, end_lng))
    
    return path

api_key = 'YOUR_GOOGLE_MAPS_API_KEY'
start_location = 'START_LOCATION_ADDRESS_OR_COORDINATES'
end_location = 'END_LOCATION_ADDRESS_OR_COORDINATES'
path_coordinates = get_path_coordinates(api_key, start_location, end_location)
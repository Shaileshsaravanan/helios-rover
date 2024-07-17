import requests

def get_current_location(api_key):
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyDEUUQO07v43cdP8wmbFPaaJTI9wLXdWRc" 
    response = requests.post(url)
    
    if response.status_code == 200:
        location_data = response.json()
        lat = location_data['location']['lat']
        lng = location_data['location']['lng']
        accuracy = location_data['accuracy']
        return lat, lng, accuracy
    else:
        return None, None, None

if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    lat, lng, accuracy = get_current_location(api_key)
    if lat and lng:
        print(f"Latitude: {lat}, Longitude: {lng}, Accuracy: {accuracy} meters")
    else:
        print("Failed to get location.")

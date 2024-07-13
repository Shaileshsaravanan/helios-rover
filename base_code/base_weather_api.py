import requests
import json

# Replace with your Tomorrow.io API Key
API_KEY = 'YOUR_API_KEY_HERE'

# Set the base URL for the Tomorrow.io API
BASE_URL = 'https://api.tomorrow.io/v4/timelines'

# Define the latitude and longitude of the location
latitude = 37.7749  # Example: San Francisco
longitude = -122.4194  # Example: San Francisco

# Define the parameters for the API request
params = {
    'location': f'{latitude},{longitude}',
    'fields': 'temperature_2m_max,temperature_2m_min,precipitation_probability_mean,precipitation_sum,wind_speed_10m_max',
    'timesteps': '1d',
    'startTime': '2024-07-13T00:00:00Z',  # Replace with the desired start date
    'endTime': '2024-08-12T00:00:00Z',  # Replace with the desired end date
    'apikey': API_KEY
}

# Make the API request
response = requests.get(BASE_URL, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    data = response.json()
    
    # Extract the weather data
    weather_data = data['data']['timelines'][0]['intervals']
    
    # Print the weather data
    for day in weather_data:
        date = day['startTime']
        temp_max = day['values']['temperature_2m_max']
        temp_min = day['values']['temperature_2m_min']
        precip_prob = day['values']['precipitation_probability_mean']
        precip_sum = day['values']['precipitation_sum']
        wind_speed = day['values']['wind_speed_10m_max']
        
        print(f"Date: {date}")
        print(f"Max Temperature: {temp_max} °C")
        print(f"Min Temperature: {temp_min} °C")
        print(f"Precipitation Probability: {precip_prob} %")
        print(f"Precipitation Sum: {precip_sum} mm")
        print(f"Max Wind Speed: {wind_speed} m/s")
        print('-' * 40)
else:
    print(f"Failed to retrieve data: {response.status_code}")
    print(response.text)
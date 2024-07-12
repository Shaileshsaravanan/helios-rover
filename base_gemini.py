import google.generativeai as genai
import os
import base64
import json
import requests

api_key = os.getenv("API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')

def get_weather_data(location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={location}&days=30"
    response = requests.get(url)
    data = response.json()
    
    forecast = []
    for day in data['forecast']['forecastday']:
        forecast.append({
            "date": day['date'],
            "max_temp": day['day']['maxtemp_c'],
            "min_temp": day['day']['mintemp_c'],
            "humidity": day['day']['avghumidity'],
            "precipitation": day['day']['totalprecip_mm']
        })
    return forecast

def process_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image_base64_str = base64.b64encode(image_data).decode('utf-8')
    prompt = {
        "image": {
            "base64": image_base64_str
        },
        "query": "Determine the state of the plant and calculate its percentage yield."
    }
    response = model.generate_content(prompt)
    result = response.text

    try:
        result_data = json.loads(result)
        plant_state = result_data.get('state', 'Unknown')
        percentage_yield = result_data.get('percentage_yield', 0.0)
    except json.JSONDecodeError:
        plant_state = "Error in response"
        percentage_yield = 0.0

    return plant_state, percentage_yield

def get_environmental_data(location):
    environmental_data = {
        "soil_moisture": 45.0,
        "humidity": 55.0,
        "temperature": 22.0
    }
    return environmental_data

def process_images(base64_images, location):
    weather_data = get_weather_data(location)
    environmental_data = get_environmental_data(location)

    results = []
    for base64_image in base64_images:
        state, yield_percentage = process_image(base64_image)
        results.append({
            "state": state,
            "percentage_yield": yield_percentage,
            "soil_moisture": environmental_data["soil_moisture"],
            "humidity": environmental_data["humidity"],
            "temperature": environmental_data["temperature"],
            "weather_forecast": weather_data
        })
    return results

if __name__ == "__main__":
    base64_images = [
        "<base64_image_1>",
        "<base64_image_2>",
        "<base64_image_3>"
    ]

    location = "New York"
    results = process_images(base64_images, location)
    
    for idx, result in enumerate(results):
        print(f"Image {idx+1}:")
        print(f"  State - {result['state']}")
        print(f"  Percentage Yield - {result['percentage_yield']:.2f}%")
        print(f"  Soil Moisture - {result['soil_moisture']}%")
        print(f"  Humidity - {result['humidity']}%")
        print(f"  Temperature - {result['temperature']}°C")
        print("  Weather Forecast (next 30 days):")
        for day in result["weather_forecast"]:
            print(f"    Date: {day['date']}, Max Temp: {day['max_temp']}°C, Min Temp: {day['min_temp']}°C, Humidity: {day['humidity']}%, Precipitation: {day['precipitation']}mm")
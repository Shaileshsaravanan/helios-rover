from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import os
import google.generativeai as genai

app = Flask(__name__)
 
user_location = {'latitude': 'none', 'longitude': 'none'}
load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_response(text):
    try:
        response = model.generate_content(text)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

@app.route('/')
def home():
    global user_location
    return render_template('dashboard.html', user_location=user_location)

@app.route('/user/location', methods=['POST'])
def user_location_get():
    global user_location
    data = request.get_json()
    print(data)
    user_location = data
    return jsonify({'status': 'Updated'}), 200

@app.route('/data/coordinates', methods=['POST'])
def data_coordinates():
    data = request.get_json()
    print(data)
    lat = data['latitude']
    lon = data['longitude']
    url = f'http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHERMAP_API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data
       

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/data/all', methods=['POST']) 
def all_data():
    data = request.get_json()
    lat = data['latitude']
    lon = data['longitude']
    response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation_probability,precipitation,rain,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,evapotranspiration,et0_fao_evapotranspiration,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,wind_direction_10m,wind_direction_80m,wind_direction_120m,wind_direction_180m,wind_gusts_10m,temperature_80m,temperature_120m,temperature_180m,soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm,uv_index,sunshine_duration,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance,terrestrial_radiation&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&past_days=92&forecast_days=16')
    return response.json()

@app.route('/data/soil', methods=['POST']) 
def soil_data():
    data = request.get_json()
    lat = data['latitude']
    lon = data['longitude']
    response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,rain,showers,snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,temperature_80m,temperature_120m,temperature_180m,soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm&past_days=92&forecast_days=16')
    return response.json()

@app.route('/data/current_weather', methods=['POST'])
def current_weather():
    data = request.get_json()
    print(data)
    lat = data['latitude']
    lon = data['longitude']
    response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m&forecast_days=1')
    return response.json()

@app.route('/gemini/weather/one_liner', methods=['POST'])
def one_liner():
    data = request.get_json()
    print(data)
    prompt = f"give me a one liner like this: Today is a partly sunny day! for the following weather data:{data}, do not include temperature or any metrics or data"
    response = generate_response(prompt)
    return response

if __name__ == '__main__':
    app.run(debug=True)

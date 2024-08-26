from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import json
import requests
import os
import google.generativeai as genai

app = Flask(__name__)
 
user_location = {"country": "IN", "lat": 13.0617012, "local_names": {"kn": "ಹೊಸಕೋಟೆ ತಾಲೂಕು"}, "lon": 77.8455574403261, "name": "Hosakote taluk", "state": "Karnataka"}
farm_and_path = {
  "type": "FeatureCollection",
  "features": [
    {
      "id": "fcd535edd036063e6b4dba5c47426cbd",
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              77.72714634818897,
              13.08841182555254
            ],
            [
              77.72678868010553,
              13.087117852275114
            ],
            [
              77.72743588901767,
              13.086222020640875
            ],
            [
              77.72930087259698,
              13.083683813331803
            ],
            [
              77.73026317006202,
              13.084646012642906
            ],
            [
              77.73049309954342,
              13.085176879619937
            ],
            [
              77.73125101524425,
              13.086163957367134
            ],
            [
              77.73188119234487,
              13.086902189401584
            ],
            [
              77.73200041503935,
              13.087242273077607
            ],
            [
              77.72714634818897,
              13.08841182555254
            ]
          ]
        ],
        "type": "Polygon"
      }
    },
    {
      "id": "c9318d8b7c9c7e62932ec3b358535793",
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            77.72718015440768,
            13.088367301561703
          ],
          [
            77.72882745689867,
            13.087969881314066
          ],
          [
            77.72877873815662,
            13.087800829224136
          ],
          [
            77.72714057042884,
            13.088189352276885
          ],
          [
            77.72708271692107,
            13.088023266160434
          ],
          [
            77.72869652527783,
            13.087631777016966
          ],
          [
            77.72863562684898,
            13.087447895539043
          ],
          [
            77.72704008802106,
            13.087857179933224
          ],
          [
            77.72699441420013,
            13.087711854392651
          ],
          [
            77.72858386318455,
            13.087278843089805
          ],
          [
            77.72851687491362,
            13.087083098004612
          ],
          [
            77.72695483022119,
            13.087513143821056
          ],
          [
            77.72691130300797,
            13.087323904843515
          ],
          [
            77.72845758484084,
            13.08687809326328
          ],
          [
            77.7283709930582,
            13.086631089201816
          ],
          [
            77.72688656249903,
            13.087107023635454
          ],
          [
            77.72710304195544,
            13.086787725952135
          ],
          [
            77.72828523084138,
            13.086454808172363
          ],
          [
            77.72818171417259,
            13.086118713636651
          ],
          [
            77.72737932850163,
            13.086381903340182
          ],
          [
            77.72756362444039,
            13.086157516063778
          ],
          [
            77.72810581412438,
            13.085980886949585
          ],
          [
            77.72880788749353,
            13.087867775726579
          ],
          [
            77.73005601792732,
            13.087563848309742
          ],
          [
            77.72990000162412,
            13.087095292805898
          ],
          [
            77.72869087526584,
            13.087449875430593
          ],
          [
            77.72856086167826,
            13.087044638103023
          ],
          [
            77.72973098396108,
            13.08670271858989
          ],
          [
            77.72945795542881,
            13.08620883401106
          ],
          [
            77.7283528399397,
            13.086563417912345
          ],
          [
            77.72811881548284,
            13.086006214410972
          ],
          [
            77.72917969215712,
            13.085599530623725
          ],
          [
            77.73043487865158,
            13.085238107586846
          ],
          [
            77.73075576247322,
            13.085863210508805
          ],
          [
            77.72950788094619,
            13.086245217070399
          ],
          [
            77.7297931110104,
            13.086766134154303
          ],
          [
            77.73111230005202,
            13.086418856220746
          ],
          [
            77.73171841393588,
            13.08697450068
          ],
          [
            77.73004268731546,
            13.08746068855362
          ],
          [
            77.72893742081965,
            13.084508818839879
          ],
          [
            77.72797476935693,
            13.085793754706984
          ],
          [
            77.73000703355626,
            13.084925555530504
          ],
          [
            77.72925830464118,
            13.084265722111695
          ],
          [
            77.72897307457885,
            13.084543546923754
          ]
        ],
        "type": "LineString"
      }
    }
  ]
}

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

@app.route('/ai_rec')
def ai_rec():
    return render_template('ai_rec.html')

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
    return render_template('map.html', user_location=user_location)

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

@app.route('/gemini/weather/oneliner', methods=['POST'])
def one_liner():
    data = request.get_json()
    print(data)
    prompt = f"give me a one liner like this: Today is a partly sunny day! for the following weather data:{data}, do not include temperature or any metrics or data"
    response = generate_response(prompt)
    print(response)
    return response

@app.route('/save/geojson', methods=['POST'])
def save_geojson():
    data = request.get_json()
    print(json.dumps(data))
    return jsonify({'status': 'Saved'}), 200

@app.route('/weather')
def weather():
    data = requests.get('https://api.open-meteo.com/v1/forecast?longitude=77.72909040403465&latitude=13.086198916771934&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation_probability,precipitation,rain,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,evapotranspiration,et0_fao_evapotranspiration,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,wind_direction_10m,wind_direction_80m,wind_direction_120m,wind_direction_180m,wind_gusts_10m,temperature_80m,temperature_120m,temperature_180m,soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm,uv_index,sunshine_duration,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance,terrestrial_radiation&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&past_days=92&forecast_days=16')
    print(data.json())
    prompt = f"{data.json()}, this is the weather data for my farmalnd,  in a ocnversational manner, tell me the expected yield, i am aware that yit may not be accurate, but just give me an estimate, and give me other data useful, also give me tips for growing my bitter guard and german tunip crops, do not mention things you cant do, no unnessary information, just the things i've asked, just give me 3estimated yield based on what you think of the temperature and all the other data provided to you, also provide a weather analysis, do not give me any warnings or alerts, i am aware of the outputs and teh accuracy, you have to just give me the data and analysis and nothing more."
    return render_template('weather.html', prompt=prompt)

@app.route('/gemini/api', methods=['POST'])
def gemini_api():
    prompt = request.get_json()
    prompt = prompt['prompt']
    response = generate_response(prompt)
    return response

@app.route('/prev')
def prev():
    return render_template('prev.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)

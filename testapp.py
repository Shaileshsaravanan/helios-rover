from flask import Flask, request, render_template
from fastkml import kml
from io import StringIO

app = Flask(__name__)

def move_rover_to_coordinate(lat, lng):
    #to be implemented later
    print(f"Moving rover to latitude: {lat}, longitude: {lng}")

def process_kml(kml_string):
    k = kml.KML()
    k.from_string(kml_string)
    
    for feature in k.features():
        for placemark in feature.features():
            if hasattr(placemark.geometry, 'geoms'):
                for line in placemark.geometry.geoms:
                    coords = list(line.coords)
                    for coord in coords:
                        lng, lat, _ = coord
                        move_rover_to_coordinate(lat, lng)
            else:
                coords = list(placemark.geometry.coords)
                for coord in coords:
                    lng, lat, _ = coord
                    move_rover_to_coordinate(lat, lng)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_kml', methods=['POST'])
def process_kml_route():
    kml_data = request.data.decode('utf-8')
    process_kml(kml_data)
    return "KML processed successfully."

if __name__ == '__main__':
    app.run(debug=True)
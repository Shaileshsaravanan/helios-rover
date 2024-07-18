#include <ESP8266WiFi.h>

// Replace with your own SSID and Password
const char* ssid = "helios.";
const char* password = "helios";

WiFiServer server(80);

void setup() {
    Serial.begin(115200);

    // Set up the ESP8266 as an Access Point
    WiFi.softAP(ssid, password);

    // Start the server
    server.begin();
    Serial.println("Access Point started");

    // Print the IP address
    Serial.print("AP IP Address: ");
    Serial.println(WiFi.softAPIP());

    // Initialize the LED (assuming the LED is connected to GPIO2)
    pinMode(2, OUTPUT);
    digitalWrite(2, LOW); // Start with LED off
}

void loop() {
    WiFiClient client = server.available();
    if (client) {
        Serial.println("New Client Connected");

        // Read the HTTP request
        String request = "";
        while (client.connected()) {
            if (client.available()) {
                char c = client.read();
                request += c;
                if (request.endsWith("\r\n\r\n")) {
                    break;
                }
            }
        }

        // Process the request
        Serial.println("Request: " + request);

        // Prepare the response
        String response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n";
        response += "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>ESP8266 Web Server</title></head><body>";
        response += "<h1>Welcome to the ESP8266 Web Server!</h1>";
        response += "<p>LED State: " + String(digitalRead(2) ? "ON" : "OFF") + "</p>";
        response += "<form action='/led_on' method='GET'><button type='submit'>Turn LED ON</button></form>";
        response += "<form action='/led_off' method='GET'><button type='submit'>Turn LED OFF</button></form>";
        response += "</body></html>";

        // Send the response
        client.print(response);

        // Check for GET requests and handle commands
        if (request.indexOf("/led_on") != -1) {
            digitalWrite(2, HIGH);  // Turn on the LED
        } else if (request.indexOf("/led_off") != -1) {
            digitalWrite(2, LOW);   // Turn off the LED
        }

        // Close the connection
        client.stop();
        Serial.println("Client Disconnected");
    }
}
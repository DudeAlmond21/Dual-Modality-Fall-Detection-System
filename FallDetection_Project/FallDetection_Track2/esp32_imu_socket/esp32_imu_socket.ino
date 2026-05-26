#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>

// ── CONFIGURE THESE ──────────────────────────
const char* ssid     = "Abisheck's iPhone";
const char* password = "12345678";
const char* serverIP = "172.20.10.4";  // e.g. "192.168.1.108"
const int   port     = 80;
// ─────────────────────────────────────────────

MPU6050 mpu;
WiFiClient client;

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22);

    mpu.initialize();
    Serial.println("MPU6050 initialized!");

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");
    Serial.println(WiFi.localIP());

    Serial.print("Connecting to server...");
    while (!client.connect(serverIP, port)) {
        Serial.print(".");
        delay(500);
    }
    Serial.println("\nConnected to server!");
}

void loop() {
    if (!client.connected()) {
        Serial.println("Disconnected! Reconnecting...");
        while (!client.connect(serverIP, port)) {
            delay(500);
        }
    }

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert accelerometer to m/s²
    float fax = (ax / 16384.0) * 9.81;
    float fay = (ay / 16384.0) * 9.81;
    float faz = (az / 16384.0) * 9.81;

    // Convert gyroscope to rad/s
    float fgx = (gx / 131.0) * (PI / 180.0);
    float fgy = (gy / 131.0) * (PI / 180.0);
    float fgz = (gz / 131.0) * (PI / 180.0);

    // Remap axes to match training data orientation
    // Training data: gravity on Y axis (negative ~-9.8)
    // Our sensor:    gravity on Z axis (positive ~+9.8)
    float send_ax = fax;
    float send_ay = -faz;  // Z → Y with sign flip
    float send_az = fay;   // Y → Z

    // Send in exact format: !ax,ay,az,gx,gy,gz@
    String data = "!" + String(send_ax, 4) + ","
                      + String(send_ay, 4) + ","
                      + String(send_az, 4) + ","
                      + String(fgx, 4) + ","
                      + String(fgy, 4) + ","
                      + String(fgz, 4) + "@";

    client.print(data);
    Serial.println(data);
    delay(47);  // ~21Hz
}
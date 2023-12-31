from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import MQTTSNclient
import json
import numpy as np
import threading

# Global list for storing temperature and pressure data
temperature_data = []
filtered_temperature_data = []
pressure_data = []
filtered_pressure_data = []

# Shared data structure and lock for synchronized access
combined_data = {
    "temperature": None,
    "pressure": None,
    "site": "Grenoble"  # Adding site name
}
data_lock = threading.Lock()

def extract_and_store_data(message):
    try:
        jsonP = json.loads(message)
        if 'temperature' in jsonP:
            temperature_data.append(float(jsonP['temperature']))
        if 'pressure' in jsonP:
            pressure_data.append(float(jsonP['pressure']))
        print("Data extracted: Temperature and Pressure")  # Debugging print
    except Exception as e:
        print("Error in extract_and_store_data:", e)

def median_filter(data):
    return np.median(data)

def calculate_average(data):
    return sum(data) / len(data) if data else 0

def process_temperature_data():
    global combined_data
    while True:
        if len(temperature_data) >= 5:
            filtered_value = median_filter(temperature_data[:5])
            del temperature_data[:5]
            with data_lock:
                combined_data["temperature"] = filtered_value
                publish_combined_data()

def process_pressure_data():
    global combined_data
    while True:
        if len(pressure_data) >= 5:
            filtered_value = median_filter(pressure_data[:5])
            del pressure_data[:5]
            with data_lock:
                combined_data["pressure"] = filtered_value
                publish_combined_data()

def publish_combined_data():
    global combined_data
    if combined_data["temperature"] is not None and combined_data["pressure"] is not None:
        
        topicName = "Grenoble/Data"
        message = json.dumps(combined_data)
        MQTTClient.publish("Grenoble/Data", message, 1)  # Example topic
        print("Outgoing : ",topicName, message)
        # Reset the values after publishing
        combined_data["temperature"] = None
        combined_data["pressure"] = None

class Callback:
    def messageArrived(self, topicName, payload, qos, retained, msgid):
        message = payload.decode("utf-8")
        data_thread = threading.Thread(target=extract_and_store_data, args=(message,))
        data_thread.start()
        jsonP = json.loads(message)
        json_topic = f"sensor/node{jsonP['node']}"
        new_string_variable = f"{json_topic}"

        if json_topic in topics_to_subscribe:
            print(f"Received message from subscribed topic: {json_topic}")
            topicName = new_string_variable
            print("Incomming : ",topicName, message)
            #MQTTClient.publish(topicName, message, qos)
        else:
            print(f"Received message from an unsubscribed topic: {json_topic}")

        return True

# MQTT broker configuration
path = "/home/root/IoT-Mini-Project-2-Data-Engineering-in-IoT-Pipeline/certs/"
MQTTClient = AWSIoTMQTTClient("MQTTSNbridge")
MQTTSNClient = MQTTSNclient.Client("bridge", port=1885)

MQTTClient.configureEndpoint("a5hi9k1blxeai-ats.iot.us-east-1.amazonaws.com", 8883)
MQTTClient.configureCredentials(path+"AmazonRootCA1.pem", 
                                path+"cb9f87c8d846f3c7aec8b3deee46ee53cb2a7939a1972ff7c812db53ce3cc041-private.pem.key", 
                                path+"cb9f87c8d846f3c7aec8b3deee46ee53cb2a7939a1972ff7c812db53ce3cc041-certificate.pem.crt")

MQTTClient.configureOfflinePublishQueueing(-1)
MQTTClient.configureDrainingFrequency(2)
MQTTClient.configureConnectDisconnectTimeout(10)
MQTTClient.configureMQTTOperationTimeout(5)

MQTTSNClient.registerCallback(Callback())

# Connect to the clients
MQTTClient.connect()
MQTTSNClient.connect()

# Define topics for subscription
topics_to_subscribe = ["sensor/node1", "sensor/node2", "sensor/node3"]
for topic in topics_to_subscribe:
    MQTTSNClient.subscribe(topic)

# Start the data processing threads
temp_thread = threading.Thread(target=process_temperature_data)
temp_thread.start()

press_thread = threading.Thread(target=process_pressure_data)
press_thread.start()

# Keep the main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    # Handle any cleanup here
    print("Shutting down...")

# Disconnect from the clients
MQTTSNClient.disconnect()
MQTTClient.disconnect()
print("Disconnected from MQTT clients.")

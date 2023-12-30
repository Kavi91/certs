from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import MQTTSNclient
import json
import numpy as np

# Global list for storing temperature data
temperature_data = []
filtered_temperature_data = []

def extract_and_store_temperature(message):
    try:
        jsonP = json.loads(message)
        if 'temperature' in jsonP:
            # Convert temperature to a float before appending
            temp_value = float(jsonP['temperature'])
            temperature_data.append(temp_value)
    except json.JSONDecodeError:
        print("Error decoding JSON")
    except ValueError:
        print("Invalid temperature value")


def median_filter(data):
    return np.median(data)

# Add a function to calculate the average
def calculate_average(data):
    return sum(data) / len(data) if data else 0

class Callback:
    def messageArrived(self, topicName, payload, qos, retained, msgid):
        message = payload.decode("utf-8")
        
        # Extract and store temperature data
        extract_and_store_temperature(message)

        jsonP = json.loads(message)
        json_topic = f"sensor/node{jsonP['node']}"
        new_string_variable = f"{json_topic}"

        if json_topic in topics_to_subscribe:
            print(f"Received message from subscribed topic: {json_topic}")
            topicName = new_string_variable
            print(topicName, message)
            MQTTClient.publish(topicName, message, qos)
        else:
            print(f"Received message from an unsubscribed topic: {json_topic}")

        return True

# Path that indicates the certificates position
path = "/home/root/IoT-Mini-Project-2-Data-Engineering-in-IoT-Pipeline/certs/"

# Configure the access with the AWS MQTT broker
MQTTClient = AWSIoTMQTTClient("MQTTSNbridge")
MQTTSNClient = MQTTSNclient.Client("bridge", port=1885)

MQTTClient.configureEndpoint("a5hi9k1blxeai-ats.iot.us-east-1.amazonaws.com", 8883)
MQTTClient.configureCredentials(path+"AmazonRootCA1.pem",
                                path+"cb9f87c8d846f3c7aec8b3deee46ee53cb2a7939a1972ff7c812db53ce3cc041-private.pem.key",
                                path+"cb9f87c8d846f3c7aec8b3deee46ee53cb2a7939a1972ff7c812db53ce3cc041-certificate.pem.crt")

# Configure the MQTT broker
MQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
MQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
MQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
MQTTClient.configureMQTTOperationTimeout(5)  # 5 sec

# Register the callback
MQTTSNClient.registerCallback(Callback())

# Connect to the clients
MQTTClient.connect()
MQTTSNClient.connect()

# Define topics for subscription
topics_to_subscribe = ["sensor/node1", "sensor/node2", "sensor/node3"]

# Subscribe to the topics
for topic in topics_to_subscribe:
    MQTTSNClient.subscribe(topic)

# Infinite loop to keep the script running
while True:
    # Check if there are 20 readings in the list
    if len(temperature_data) >= 20:
        # Apply median filter to the temperature data
        filtered_value = median_filter(temperature_data)
        filtered_temperature_data.append(filtered_value)
        print("Filtered Temperature Value:", filtered_value)

        # Clear the original data list after filtering
        temperature_data.clear()
        # Calculate and print the average of the filtered data
        if len(filtered_temperature_data) % 20 == 0:  # For example, calculate every 20 filtered values
            average_temp = calculate_average(filtered_temperature_data)
            print("Average Filtered Temperature Value:", average_temp)

# Disconnect from the clients (this part will not be reached)
MQTTSNClient.disconnect()
MQTTClient.disconnect()

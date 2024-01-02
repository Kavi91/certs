import random
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy import stats
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def generate_custom_noise():
    noise_prob = random.uniform(0, 1)
    if noise_prob < 0.2:  # Increase the probability (e.g., 20%) for more frequent noise
        # Generate a missing value (NaN) with a 20% probability
        return np.nan
    elif noise_prob < 0.4:
        # Generate a larger positive outlier with a 20% probability
        return random.uniform(45.0, 50.0)  # Increase the range for significant impact
    elif noise_prob < 0.6:
        # Generate a larger negative outlier with a 20% probability
        return random.uniform(35.0, 38.0)  # Increase the range for significant impact
    else:
        # Generate a value farther from 0 with a 40% probability
        return random.uniform(-5.0, 5.0)  # Increase the range for significant impact

def generate_random_temperature():
    noise = generate_custom_noise()
    if not np.isnan(noise):
        base_temperature = 40.0  # Set the base temperature to 40 degrees
        return base_temperature + noise
    else:
        return noise

# Create a DataFrame to store the original temperature data
original_data_df = pd.DataFrame(columns=['Temperature'])

# Lists to store data for plotting
time_points = []
smoothed_temperatures = []

# Linear regression coefficients
linear_coefficients = {'slope': [], 'intercept': []}

# Set the update interval (5 seconds)
update_interval = 5

# Generate and save 100 raw data points to a CSV file
raw_data_file = 'raw_temperature_data.csv'
num_data_points = 100

with open(raw_data_file, 'w') as file:
    file.write('Temperature\n')
    for _ in range(num_data_points):
        random_temperature = generate_random_temperature()
        file.write(f'{random_temperature}\n')

# Read the raw data from the CSV file
raw_data = pd.read_csv(raw_data_file)

# Create a subplot for the regression graph
plt.ion()  # Turn on interactive mode for continuous updating
fig, ax = plt.subplots(figsize=(10, 6))

# Running average variables
running_sum = 0.0
running_count = 0

# Data preprocessing loop
for idx, row in raw_data.iterrows():
    # Add the new data point to the DataFrame
    new_data = pd.DataFrame({'Temperature': [row['Temperature']]})
    original_data_df = pd.concat([original_data_df, new_data], ignore_index=True)
    
    # Impute missing values using forward fill (ffill)
    original_data_df['Temperature'].ffill(inplace=True)
    
    # Calculate the Z-score for each temperature reading
    z_scores = np.abs(stats.zscore(original_data_df['Temperature']))
    
    # Define a Z-score threshold for outlier detection
    z_score_threshold = 2.0  # Adjust as needed
    
    # Mark outliers based on the Z-score threshold
    original_data_df['Outlier'] = z_scores > z_score_threshold
    
    # Filter out outliers
    original_data_df = original_data_df[~original_data_df['Outlier']]
    
    # Apply a median filter to the filtered data with a larger kernel_size
    original_data_df['Smoothed_Temperature'] = medfilt(original_data_df['Temperature'], kernel_size=9)
    
    # Get the most recent time and smoothed temperature
    current_time = time.time()
    current_smoothed_temperature = original_data_df['Smoothed_Temperature'].iloc[-1]
    
    # Fit a linear regression model to smoothed temperature data
    if len(time_points) >= 2:
        time_diff = np.array(time_points) - time_points[0]  # Time relative to the start
        coefficients = np.polyfit(time_diff, smoothed_temperatures, 1)  # Fit a linear model
        linear_coefficients['slope'].append(coefficients[0])
        linear_coefficients['intercept'].append(coefficients[1])
    
    # Update the plots continuously
    time_points.append(current_time)
    smoothed_temperatures.append(current_smoothed_temperature)
    
    # Update running average
    running_sum += current_smoothed_temperature
    running_count += 1
    
    ax.clear()
    ax.plot(time_points, smoothed_temperatures, label='Smoothed Temperature', marker='x', linestyle='-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Regression Graph')
    plt.pause(0.01)

# Final regression graph
time_diff = np.array(time_points) - time_points[0]  # Time relative to the start
coefficients = np.polyfit(time_diff, smoothed_temperatures, 1)  # Fit a linear model

# Calculate the final smoothed temperature as the running average
final_smoothed_temperature = running_sum / running_count

# Print the final smoothed temperature
print(f"Final Smoothed Temperature: {final_smoothed_temperature:.2f} °C")

# Print the final linear regression coefficients
print(f"Final Linear Regression Coefficients (Smoothed Temperature): Slope = {coefficients[0]:.4f}, Intercept = {coefficients[1]:.4f}")

# Save the regression graph as an image (optional)
plt.savefig('regression_graph.png')

# Display the final regression graph
plt.ioff()  # Turn off interactive mode
plt.show()

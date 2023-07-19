import pandas as pd
from joblib import load
import timeit
import logging
import serial

# Mapping for rating labels
rating_labels = {
    1: "sangat buruk",
    2: "buruk",
    3: "biasa",
    4: "baik",
    5: "sangat baik"
}

# Chunk size for iterating over data
chunk_size = 1000

# Configure logging
log_filename = 'error_log.txt'
logging.basicConfig(filename=log_filename, level=logging.ERROR)

# Load the previously trained model
model_filename = '../Training/water_quality_model.pkl'
model = load(model_filename)

# Create an empty list to store the results
result_data = []

# Establish connection with Arduino
arduino_port = 'COM3'  # Replace with the appropriate port
arduino_baudrate = 9600  # Set the baudrate to match your Arduino configuration
arduino = serial.Serial(arduino_port, arduino_baudrate)

# Read input data from Arduino
def read_data_from_arduino():
    # Read data from Arduino
    arduino_data = []
    while len(arduino_data) < chunk_size:
        line = arduino.readline().decode().strip()
        if line:
            arduino_data.append(float(line))
    return pd.DataFrame(arduino_data, columns=['SensorData'])

# Predict water quality based on Arduino data
def predict_water_quality(input_data):
    # Calculate the number of iterations needed based on the chunk size
    num_iterations = (len(input_data) + chunk_size - 1) // chunk_size

    # Continuous data processing loop
    try:
        for i in range(num_iterations):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, len(input_data))
            chunk = input_data.iloc[start_index:end_index]

            # Predict water quality ratings and measure elapsed time
            elapsed_time = timeit.timeit(lambda: model.predict(chunk), number=1)

            # Map predictions to desired labels
            y_pred = model.predict(chunk)
            y_pred_mapped = [rating_labels.get(prediction, "unknown") for prediction in y_pred]

            # Display predictions
            for prediction in y_pred_mapped:
                print('Prediksi kualitas air:', prediction)

            # Add predictions to list
            result_data.extend(y_pred_mapped)

            # Calculate the remaining time to sleep for 100 milliseconds
            remaining_time = max(0.1 - elapsed_time, 0)
            time.sleep(remaining_time)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")

    # Convert the result list to a DataFrame
    result_df = pd.DataFrame({'Prediksi': result_data})

    # Save the result data to an Excel file
    excel_filename = 'water_quality_predictions.xlsx'
    excel_counter = 1
    while os.path.exists(excel_filename):
        excel_filename = f'water_quality_predictions_{excel_counter}.xlsx'
        excel_counter += 1

    result_df.to_excel(excel_filename, index=False)
    print(f"Data telah disimpan ke file {excel_filename}.")

# Read data from Arduino
arduino_data = read_data_from_arduino()

# Call the function to predict water quality
predict_water_quality(arduino_data)

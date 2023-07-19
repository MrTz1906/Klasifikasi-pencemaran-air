import os
import pandas as pd
import pickle
import serial
import time

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

# Memuat model yang telah dilatih sebelumnya
model_filename = 'water_quality_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Establish serial communication with Arduino
arduino_port = 'COM3'  # Replace with the correct port
baud_rate = 9600
arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Allow time for Arduino to reset

# Create an empty DataFrame to store the results
result_data = pd.DataFrame(columns=['Prediksi'])

# Iterasi melalui data dalam chunk
while True:
    # Membaca data dari Arduino
    data = arduino.readline().decode().strip()
    if data:
        # Memisahkan data menjadi fitur input (X)
        values = data.split(',')
        conductivity = float(values[0])
        temperature = float(values[1])
        turbidity = float(values[2])
        total_dissolved_solids = float(values[3])

        # Membuat DataFrame dengan data dari Arduino
        input_data = pd.DataFrame({'conductivity': [conductivity],
                                   'temperature': [temperature],
                                   'turbidity': [turbidity],
                                   'total_dissolved_solids': [total_dissolved_solids]})

        # Memprediksi rating kualitas air
        y_pred = model.predict(input_data)

        # Memetakan prediksi ke label yang diinginkan
        y_pred_mapped = [rating_labels.get(prediction, "unknown") for prediction in y_pred]

        # Menampilkan hasil prediksi
        prediction = y_pred_mapped[0]
        print('Prediksi kualitas air:', prediction)

        # Menambahkan prediksi ke DataFrame
        result_data = result_data.append({'Prediksi': prediction}, ignore_index=True)

    # Check if spacebar is pressed to break the loop
    if os.name == 'nt':
        import msvcrt
        if msvcrt.kbhit() and msvcrt.getch() == b' ':
            print('Proses prediksi dihentikan.')
            break
    else:
        import select
        if select.select([sys.stdin,],[],[],0.01)[0]:
            break

    # Delay for 100 ms
    time.sleep(0.1)

# Close serial communication with Arduino
arduino.close()

# Save the result data to an Excel file
excel_filename = 'water_quality_predictions.xlsx'
excel_counter = 1
while os.path.exists(excel_filename):
    excel_filename = f'water_quality_predictions_{excel_counter}.xlsx'
    excel_counter += 1

with pd.ExcelWriter(excel_filename) as writer:
    result_data.to_excel(writer, sheet_name='Hasil', index=False)

print('Prediksi kualitas air berhasil dilakukan. Hasil disimpan dalam file:', excel_filename)

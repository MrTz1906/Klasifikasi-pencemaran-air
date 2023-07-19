import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import serial
import os
import joblib

# Membuat koneksi serial dengan Arduino
ser = serial.Serial('COM3', 9600)  # Ganti 'COM3' dengan port serial Arduino yang sesuai

# Membuat daftar kosong untuk menyimpan data sensor
conductivity_list = []
temperature_list = []
turbidity_list = []
total_dissolved_solids_list = []

# Set durasi dan interval pencatatan
duration = 300  # Durasi dalam detik (5 menit)
interval = 0.1  # Interval antara pembacaan dalam detik (100 ms)

# Menghitung jumlah iterasi
iterations = int(duration / interval)

# Merekam data selama durasi yang ditentukan
for _ in range(iterations):
    # Membaca satu baris data dari Arduino
    data = ser.readline().decode().strip()

    # Memisahkan data menjadi nilai-nilai individual
    values = data.split(',')

    # Ekstraksi nilai-nilai sensor
    conductivity = float(values[0])
    temperature = float(values[1])
    turbidity = float(values[2])
    total_dissolved_solids = float(values[3])

    # Menambahkan nilai-nilai sensor ke daftar yang sesuai
    conductivity_list.append(conductivity)
    temperature_list.append(temperature)
    turbidity_list.append(turbidity)
    total_dissolved_solids_list.append(total_dissolved_solids)

    # Menunggu interval yang ditentukan
    time.sleep(interval)

# Membuat kamus dengan data sensor
data_dict = {
    'conductivity': conductivity_list,
    'temperature': temperature_list,
    'turbidity': turbidity_list,
    'total_dissolved_solids': total_dissolved_solids_list
}

# Membuat DataFrame dari kamus data
data = pd.DataFrame(data_dict)

# Load model regresi logistik yang telah dilatih sebelumnya
model_file_name = 'water_quality_model.pkl'
model = joblib.load(model_file_name)

# Memprediksi rating kualitas air untuk data uji
y_pred = model.predict(data)

# Mengubah prediksi menjadi rentang yang diinginkan (1 hingga 5)
y_pred_mapped = [max(min(int(round(prediction)), 5), 1) for prediction in y_pred]

# Membuat DataFrame dengan prediksi
output_data = pd.DataFrame({'Prediksi Rating': y_pred_mapped})

# Menyimpan prediksi ke file Excel
output_file_name = 'water_quality_predictions.xlsx'
output_data.to_excel(output_file_name, sheet_name='Hasil', index=False)

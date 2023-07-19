import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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
water_quality_rating_list = []

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

    # Ekstraksi nilai-nilai sensor dan rating kualitas air
    conductivity = float(values[0])
    temperature = float(values[1])
    turbidity = float(values[2])
    total_dissolved_solids = float(values[3])
    water_quality_rating = int(values[4])  # Anggap rating diberikan oleh Arduino

    # Menambahkan nilai-nilai sensor dan rating kualitas air ke daftar yang sesuai
    conductivity_list.append(conductivity)
    temperature_list.append(temperature)
    turbidity_list.append(turbidity)
    total_dissolved_solids_list.append(total_dissolved_solids)
    water_quality_rating_list.append(water_quality_rating)

    # Menunggu interval yang ditentukan
    time.sleep(interval)

# Membuat kamus dengan data sensor
data_dict = {
    'conductivity': conductivity_list,
    'temperature': temperature_list,
    'turbidity': turbidity_list,
    'total_dissolved_solids': total_dissolved_solids_list,
    'water_quality_rating': water_quality_rating_list
}

# Membuat DataFrame dari kamus data
data = pd.DataFrame(data_dict)

# Memisahkan data menjadi fitur input (X) dan variabel target (y)
X = data[['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids']]
y = data['water_quality_rating']

# Memisahkan data menjadi set data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat dan melatih model regresi logistik
model = LogisticRegression()
model.fit(X_train, y_train)

# Memprediksi rating kualitas air untuk data uji
y_pred = model.predict(X_test)

# Mengubah prediksi menjadi rentang yang diinginkan (1 hingga 5)
y_pred_mapped = [max(min(int(round(prediction)), 5), 1) for prediction in y_pred]

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred_mapped)

# Menghitung matriks konfusi per data
cm_per_data = confusion_matrix(y_test, y_pred_mapped)

# Membuat DataFrame dengan akurasi dan matriks konfusi per data
result_data = pd.DataFrame({'Akurasi': [accuracy]})
result_data = pd.concat([result_data, pd.DataFrame(cm_per_data)], axis=1)

# Menyimpan data hasil dan data uji ke file Excel
with pd.ExcelWriter('water_quality_predictions.xlsx') as writer:
    # Sheet Hasil
    result_data.to_excel(writer, sheet_name='Hasil', index=False)
    
    # Sheet Data Uji
    X_test_with_cm = X_test.copy()
    X_test_with_cm['water_quality_rating'] = y_test
    X_test_with_cm['Prediksi'] = y_pred_mapped
    X_test_with_cm['Klasifikasi'] = ['True Positive' if p == a else 'False Positive' for p, a in zip(y_pred_mapped, y_test)]
    X_test_with_cm.to_excel(writer, sheet_name='Data Uji', index=False)
    
    # Sheet Data Latih
    X_train_with_rating = X_train.copy()
    X_train_with_rating['water_quality_rating'] = y_train
    X_train_with_rating.to_excel(writer, sheet_name='Data Latih', index=False)

# Menyimpan model yang telah dilatih untuk penggunaan selanjutnya
model_file_name = 'water_quality_model.pkl'
joblib.dump(model, model_file_name)

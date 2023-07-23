import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Memuat data dari CSV
data = pd.read_csv('../Preproccess/water_potabilitysesuai.csv')

# Memisahkan fitur input (X) dan variabel target (y)
X = data[['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids']]
y = data['water_quality_rating']

# Memisahkan data menjadi set data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat dan melatih model regresi logistik
model = LogisticRegression()
model.fit(X_train, y_train)

# Menyimpan model yang telah dilatih dengan penamaan berulang jika sudah ada file dengan nama yang sama
model_filename = '../PostTrain/water_quality_model.pkl'
model_counter = 1
while os.path.exists(model_filename):
    model_filename = f'../PostTrain/water_quality_model_{model_counter}.pkl'
    model_counter += 1
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Memprediksi rating kualitas air untuk data uji
y_pred = model.predict(X_test)

# Memetakan prediksi ke rentang yang diinginkan (0 hingga 1)
y_pred_mapped = [max(min(int(round(prediction)), 2), 0) for prediction in y_pred]

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred_mapped)

# Menghitung matriks kebingungan (confusion matrix)
confusion = confusion_matrix(y_test, y_pred_mapped)

# Membuat DataFrame dengan akurasi dan matriks kebingungan
result_data = pd.DataFrame({'water_quality_rating': y_test.values, 'Prediksi': y_pred_mapped, 'Akurasi': accuracy})

# Menyimpan data hasil dan data uji dalam file Excel dengan penamaan berulang jika sudah ada file dengan nama yang sama
excel_filename = 'water_quality_predictions.xlsx'
excel_counter = 1
while os.path.exists(excel_filename):
    excel_filename = f'water_quality_predictions_{excel_counter}.xlsx'
    excel_counter += 1
with pd.ExcelWriter(excel_filename) as writer:
    # Lembar Hasil
    result_data.to_excel(writer, sheet_name='Hasil', index=False)

    # Lembar Data Uji
    X_test_with_cm = X_test.copy()
    X_test_with_cm['water_quality_rating'] = y_test.values
    X_test_with_cm['Prediksi'] = y_pred_mapped
    X_test_with_cm['Klasifikasi'] = ['True Positive' if p == a else 'False Positive' for p, a in zip(y_pred_mapped, y_test)]
    X_test_with_cm.to_excel(writer, sheet_name='Data Uji', index=False)

    # Lembar Data Latih
    X_train_with_rating = X_train.copy()
    X_train_with_rating['water_quality_rating'] = y_train.values
    X_train_with_rating.to_excel(writer, sheet_name='Data Latih', index=False)

    # Lembar Matriks Kebingungan (Confusion Matrix)
    # Menghitung true positive (TP), true negative (TN), false positive (FP), dan false negative (FN)
    tp = confusion[1, 1]
    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    confusion_data = pd.DataFrame({'': ['Prediksi Positif', 'Prediksi Negatif'], 'Aktual Positif': [tp, fn], 'Aktual Negatif': [fp, tn]})
    confusion_data.to_excel(writer, sheet_name='Matriks Kebingungan', index=False)

# Menampilkan pesan berhasil
print('Pemodelan dan prediksi kualitas air berhasil dilakukan. Hasil disimpan dalam file:', excel_filename)

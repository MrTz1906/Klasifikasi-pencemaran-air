import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat data dari CSV
data = pd.read_csv('../Preproccess/water_potabilitysesuai.csv')

# Data Preprocessing
data.fillna(data.mean(), inplace=True)

# Memisahkan fitur input (X) dan variabel target (y)
X = data[['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids']]
y = data['water_quality_rating']

# Feature Scaling - Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Memisahkan data menjadi set data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Membuat dan melatih model regresi logistik dengan hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
}
model = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Menyimpan model yang telah dilatih dengan penamaan berulang jika sudah ada file dengan nama yang sama
model_filename = '../PostTrain/water_quality_model.pkl'
model_counter = 1
while os.path.exists(model_filename):
    model_filename = f'../PostTrain/water_quality_model_{model_counter}.pkl'
    model_counter += 1
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

# Memprediksi probabilitas rating kualitas air untuk data uji
y_prob = best_model.predict_proba(X_test)[:, 1]

# Memetakan probabilitas ke rentang 0 atau 1 menggunakan threshold 0.3
y_pred_binary = (y_prob >= 0.4089).astype(int)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred_binary)

# Menghitung matriks kebingungan (confusion matrix)
confusion = confusion_matrix(y_test, y_pred_binary)

# Membuat DataFrame dengan akurasi dan matriks kebingungan
result_data = pd.DataFrame({'water_quality_rating': y_test.values, 'Prediksi': y_pred_binary, 'Akurasi': accuracy})

# Plot Dot Chart
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_binary, color='blue', alpha=0.6)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Actual Water Quality Rating')
plt.ylabel('Predicted Water Quality Rating')
plt.title('Dot Chart - Actual vs. Predicted')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

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
    X_test_with_cm = pd.DataFrame(X_test, columns=X.columns)
    X_test_with_cm['water_quality_rating'] = y_test.values
    X_test_with_cm['Prediksi'] = y_pred_binary
    X_test_with_cm['Klasifikasi'] = ['True Positive' if p == a else 'False Positive' for p, a in zip(y_pred_binary, y_test)]
    X_test_with_cm.to_excel(writer, sheet_name='Data Uji', index=False)

    # Lembar Data Latih
    X_train_with_rating = pd.DataFrame(X_train, columns=X.columns)
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

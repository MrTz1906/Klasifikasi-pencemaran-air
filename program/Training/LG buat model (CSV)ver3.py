import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
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
y_pred_binary = (y_prob >= 0.3).astype(int)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred_binary)

# Menghitung matriks kebingungan (confusion matrix)
confusion = confusion_matrix(y_test, y_pred_binary)

# Membuat DataFrame dengan akurasi dan matriks kebingungan
result_data = pd.DataFrame({'water_quality_rating': y_test.values, 'Prediksi': y_pred_binary, 'Akurasi': accuracy})

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# ... Rest of the code remains the same ...

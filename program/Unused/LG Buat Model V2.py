import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
import pickle

# Load data from CSV
data = pd.read_csv('example_data.csv')

# Separate input features (X) and target variable (y)
X = data[['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids']]
y = data['water_quality_rating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a polynomial features transformer
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# Create a logistic regression model
logreg = LogisticRegression()

# Create a pipeline that applies polynomial features and logistic regression
model = make_pipeline(poly_features, logreg)

# Train the model
model.fit(X_train, y_train)

# Save the trained model with incremented filename
model_filename = '../water_quality_model.pkl'
model_counter = 1
while os.path.exists(model_filename):
    model_filename = f'water_quality_model_{model_counter}.pkl'
    model_counter += 1
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Predict the water quality ratings for test data
y_pred = model.predict(X_test)

# Map the predictions to the desired range (1 to 5)
y_pred_mapped = [max(min(int(round(prediction)), 5), 1) for prediction in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_mapped)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred_mapped)

# Create a DataFrame with accuracy and confusion matrix
result_data = pd.DataFrame({'water_quality_rating': y_test.values, 'Prediksi': y_pred_mapped, 'Akurasi': accuracy})

# Save the results and test data in an Excel file with incremented filename
excel_filename = '../water_quality_predictions.xlsx'
excel_counter = 1
while os.path.exists(excel_filename):
    excel_filename = f'water_quality_predictions_{excel_counter}.xlsx'
    excel_counter += 1
with pd.ExcelWriter(excel_filename) as writer:
    # Sheet 1: Results
    result_data.to_excel(writer, sheet_name='Hasil', index=False)

    # Sheet 2: Test Data
    X_test_with_cm = X_test.copy()
    X_test_with_cm['water_quality_rating'] = y_test.values
    X_test_with_cm['Prediksi'] = y_pred_mapped
    X_test_with_cm['Klasifikasi'] = ['True Positive' if p == a else 'False Positive' for p, a in zip(y_pred_mapped, y_test)]
    X_test_with_cm.to_excel(writer, sheet_name='Data Uji', index=False)

    # Sheet 3: Training Data
    X_train_with_rating = X_train.copy()
    X_train_with_rating['water_quality_rating'] = y_train.values
    X_train_with_rating.to_excel(writer, sheet_name='Data Latih', index=False)

    # Sheet 4: Confusion Matrix
    tp = confusion[1, 1]
    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    confusion_data = pd.DataFrame({'': ['Prediksi Positif', 'Prediksi Negatif'], 'Aktual Positif': [tp, fn], 'Aktual Negatif': [fp, tn]})
    confusion_data.to_excel(writer, sheet_name='Matriks Kebingungan', index=False)

# Print the success message with the filenames
print('Pemodelan dan prediksi kualitas air berhasil dilakukan. Hasil disimpan dalam file:', excel_filename)
print('Model disimpan dalam file:', model_filename)

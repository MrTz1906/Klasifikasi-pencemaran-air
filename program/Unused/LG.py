import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load data from CSV
data = pd.read_csv('input_data.csv')

# Split the data into input features (X) and target variable (y)
X = data[['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids']]
y = data['water_quality_rating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the water quality ratings for the test data
y_pred = model.predict(X_test)

# Map the predictions to the desired range (1 to 5)
y_pred_mapped = [max(min(int(round(prediction)), 5), 1) for prediction in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_mapped)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_mapped)

# Create a DataFrame with the accuracy and confusion matrix
result_data = pd.DataFrame({'Accuracy': [accuracy]})
result_data = pd.concat([result_data, pd.DataFrame(cm)], axis=1)

# Save the result data and output data in the Excel file
with pd.ExcelWriter('water_quality_predictions.xlsx') as writer:
    result_data.to_excel(writer, sheet_name='Results', index=False)
    output_data.to_excel(writer, sheet_name='Predictions', index=False)

# Save the trained model for later use
joblib.dump(model, 'water_quality_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib  # Ensure this is imported for saving the model

# Read the data
df = pd.read_csv("v11NumericIncidentPrediction.csv")  # Update this path if needed

# Define the feature columns and target column
feature_columns = ["Victim Age", "Overall Race", "Zip Code", "Hour", "Day of Week", "Day of Month", "Month"]
target_column = "CIBRS Offense Description"

# Split the data into features and target
X = df[feature_columns]
y = df[target_column]

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to a DataFrame for better readability
labels = np.unique(y)
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.show()

# Save the trained model with compression
compressed_model_path = 'incident_prediction_model_compressed.pkl'
joblib.dump(rf_clf, compressed_model_path, compress=3)

print(f"Model saved as {compressed_model_path} with compression.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib  # For saving the model
from sklearn.preprocessing import LabelEncoder  # For encoding categorical data

# Read the data
df = pd.read_csv("v11NumericIncidentPrediction.csv")  # Update this path if needed

# Ensure that "City" is used instead of "Zip Code"
feature_columns = ["Victim Age", "Overall Race", "City", "Hour", "Day of Week", "Day of Month", "Month"]
target_column = "CIBRS Offense Description"

# Encode the "City" column as it is categorical
label_encoder = LabelEncoder()
df["City"] = label_encoder.fit_transform(df["City"])

# Save the LabelEncoder for later use
joblib.dump(label_encoder, 'city_label_encoder.pkl')

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

# Feature Importance Chart
# Extract raw feature importance
feature_importances = rf_clf.feature_importances_
feature_names = feature_columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Group features for categorical encodings
grouped_features = {
    'Day of Week': ['Day of Week'],
    'Month': ['Month'],
    'Victim Age': ['Victim Age'],
    'Overall Race': ['Overall Race'],
    'City': ['City'],
    'Hour': ['Hour'],
    'Day of Month': ['Day of Month']
}

# Sum importance scores for grouped features
grouped_importance = {}
for group, columns in grouped_features.items():
    grouped_importance[group] = importance_df[importance_df['Feature'].isin(columns)]['Importance'].sum()

# Convert to DataFrame for plotting
grouped_importance_df = pd.DataFrame(list(grouped_importance.items()), columns=['Feature Group', 'Total Importance'])

# Sort by importance
grouped_importance_df = grouped_importance_df.sort_values(by='Total Importance', ascending=False)

# Plot the feature importance (vertical bars)
plt.figure(figsize=(10, 8))
sns.barplot(
    y='Total Importance',
    x='Feature Group',
    data=grouped_importance_df,
    palette='viridis'
)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xlabel("Feature Group")
plt.xticks(rotation=45, ha="right")  # Tilt feature group names for better readability
plt.tight_layout()
plt.show()

# Save the trained model with compression
compressed_model_path = 'incident_prediction_model_compressed.pkl'
joblib.dump(rf_clf, compressed_model_path, compress=3)

print(f"Model saved as {compressed_model_path} with compression.")

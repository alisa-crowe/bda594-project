import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the data
df = pd.read_csv("v11NumericIncidentPrediction.csv")

# Define the feature columns and target column
feature_columns = ["Victim Age", "Overall Race", "HHSA Region", "Hour", "Day of Week", "Day of Month", "Month"]
target_column = "CIBRS Offense Description"

X, y = df[feature_columns], df[target_column]

# Map categorical columns
race_map = {
    'HISPANIC': 0,
    'WHITE': 1,
    'OTHER': 2,
    'BLACK': 3,
    'ASIAN': 4,
    'UNKNOWN': 5,
    'PACIFIC ISLANDER': 6,
    'AMERICAN INDIAN': 7
}
day_of_week_map = {
    'MONDAY': 0,
    'TUESDAY': 1,
    'WEDNESDAY': 2,
    'THURSDAY': 3,
    'FRIDAY': 4,
    'SATURDAY': 5,
    'SUNDAY': 6
}

# Apply mappings to columns
if 'Overall Race' in feature_columns:
    X['Overall Race'] = X['Overall Race'].map(race_map)

if 'Day of Week' in feature_columns:
    X['Day of Week'] = X['Day of Week'].map(day_of_week_map)

# Encode 'HHSA Region' using LabelEncoder
if 'HHSA Region' in feature_columns:
    label_encoder = LabelEncoder()
    X['HHSA Region'] = label_encoder.fit_transform(X['HHSA Region'])

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Save the updated model
joblib.dump(clf, 'decision_tree_model_with_hhsa_region.pkl')
joblib.dump(label_encoder, 'hhsa_region_label_encoder.pkl')  # Save label encoder for consistent future mapping

# Evaluate the model
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Save the performance metrics
with open("model_evaluation_with_hhsa_region.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print("Evaluation metrics saved to 'model_evaluation_with_hhsa_region.txt'.")
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv("v11NumericIncidentPrediction.csv")

# Define the feature columns and target column
feature_columns = ["Victim Age", "Overall Race", "Zip Code", "Hour", "Day of Week", "Day of Month", "Month"]
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

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Save the updated model
import joblib
joblib.dump(clf, 'decision_tree_model.pkl')

print("Model training complete. Saved as 'decision_tree_model.pkl'.")

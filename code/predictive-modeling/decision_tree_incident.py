import pandas as pd
from sklearn import tree

# Read the data
df = pd.read_csv("v11NumericIncidentPrediction.csv")

# Define the feature columns and target column
feature_columns = ["Victim Age", "Overall Race", "Zip Code", "Hour", "Day of Week", "Day of Month", "Month"]
target_column = "CIBRS Offense Description"

X, y = df[feature_columns], df[target_column]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

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

# Save the updated model
import joblib
joblib.dump(clf, 'decision_tree_model.pkl')
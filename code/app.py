from flask import Flask, request, jsonify
import pandas as pd
from sklearn import tree

app = Flask(__name__)

# Load the data and train the model
df = pd.read_csv("predictive-modeling/v11NumericIncidentPrediction.csv")
feature_columns = ["Victim Age", "Overall Race", "Zip Code", "Domestic Violence Incident",
                   "Hour", "Day of Week", "Day of Month", "Month"]
target_column = "CIBRS Offense Description"
X, y = df[feature_columns], df[target_column]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
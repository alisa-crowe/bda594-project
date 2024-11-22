import pandas as pd
from sklearn import tree

# read the data
df = pd.read_csv("predictive-modeling/v11NumericIncidentPrediction.csv")

# Define the feature columns and target column
feature_columns = ["Victim Age", "Overall Race", "Zip Code", "Domestic Violence Incident",
                   "Hour", "Day of Week", "Day of Month", "Month"]
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

# Function to make predictions
def predict_incident(victim_age, overall_race, zip_code, domestic_violence_incident, hour, day_of_week, day_of_month,
                     month):
    # Map the input values to the corresponding numerical values
    overall_race = race_map[overall_race]
    day_of_week = day_of_week_map[day_of_week]

    # Create a DataFrame for the input values
    input_data = pd.DataFrame({
        'Victim Age': [victim_age],
        'Overall Race': [overall_race],
        'Zip Code': [zip_code],
        'Domestic Violence Incident': [domestic_violence_incident],
        'Hour': [hour],
        'Day of Week': [day_of_week],
        'Day of Month': [day_of_month],
        'Month': [month]
    })

    # Make a prediction
    prediction = clf.predict(input_data)
    return prediction[0]
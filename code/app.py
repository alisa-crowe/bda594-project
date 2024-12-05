from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Define the mappings
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Map the input values to the corresponding numerical values
    data['Overall Race'] = race_map[data['Overall Race']]
    data['Day of Week'] = day_of_week_map[data['Day of Week']]

    # Create a DataFrame for the input values
    input_data = pd.DataFrame({
        'Victim Age': [data['Victim Age']],
        'Overall Race': [data['Overall Race']],
        'Zip Code': [data['Zip Code']],
        'Domestic Violence Incident': [data['Domestic Violence Incident']],
        'Hour': [data['Hour']],
        'Day of Week': [data['Day of Week']],
        'Day of Month': [data['Day of Month']],
        'Month': [data['Month']]
    })

    # Make a prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
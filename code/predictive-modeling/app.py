from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS globally for all routes

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'incident_type_model.pkl')
model = joblib.load(model_path)

# Define mappings
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

hhsa_region_map = {
    'REGION_A': 0,
    'REGION_B': 1,
    'REGION_C': 2,
    'REGION_D': 3,
    'REGION_E': 4  # Add more regions as needed
}

month_map = {
    'JANUARY': 1,
    'FEBRUARY': 2,
    'MARCH': 3,
    'APRIL': 4,
    'MAY': 5,
    'JUNE': 6,
    'JULY': 7,
    'AUGUST': 8,
    'SEPTEMBER': 9,
    'OCTOBER': 10,
    'NOVEMBER': 11,
    'DECEMBER': 12
}

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Parse the input JSON
        data = request.get_json(force=True)

        # Map categorical values to numerical values
        data['Overall Race'] = race_map.get(data['Overall Race'].upper(), -1)
        data['Day of Week'] = day_of_week_map.get(data['Day of Week'].upper(), -1)
        data['HHSA Region'] = hhsa_region_map.get(data['HHSA Region'].upper(), -1)
        data['Month'] = month_map.get(data['Month'].upper(), -1)

        # Validate input
        if -1 in (data['Overall Race'], data['Day of Week'], data['HHSA Region'], data['Month']):
            return jsonify({'error': 'Invalid categorical input values'}), 400

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Victim Age': [data['Victim Age']],
            'Overall Race': [data['Overall Race']],
            'Hour': [data['Hour']],
            'Day of Week': [data['Day of Week']],
            'Day of Month': [data['Day of Month']],
            'Month': [data['Month']],
            'HHSA Region': [data['HHSA Region']]
        })

        # Predict the outcome
        prediction = model.predict(input_data)

        # Map prediction back to the class label
        prediction_label = model.named_steps['classifier'].classes_[prediction[0]]

        # Return the prediction
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Incident Predictor is Running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Heroku-assigned port
    app.run(host='0.0.0.0', port=port)

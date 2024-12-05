from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS globally for all routes

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_model.pkl')
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

        # Validate input
        if -1 in (data['Overall Race'], data['Day of Week']):
            return jsonify({'error': 'Invalid categorical input values'}), 400

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Victim Age': [data['Victim Age']],
            'Overall Race': [data['Overall Race']],
            'Zip Code': [data['Zip Code']],
            'Hour': [data['Hour']],
            'Day of Week': [data['Day of Week']],
            'Day of Month': [data['Day of Month']],
            'Month': [data['Month']]
        })

        # Predict the outcome
        prediction = model.predict(input_data)

        # Return the prediction
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Incident Predictor is Running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Heroku-assigned port
    app.run(host='0.0.0.0', port=port)

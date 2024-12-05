from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Allow CORS for the specific Google Sites domain
CORS(app, resources={r"/*": {"origins": "https://your-google-sites-domain"}})  # Replace with your actual domain

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError("Model file 'decision_tree_model.pkl' not found. Ensure it exists in the correct location.")

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

@app.route('/')
def home():
    """Root route for app health check."""
    return "Incident Predictor is Running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Route to handle prediction requests."""
    if request.method == 'OPTIONS':
        return '', 204  # Respond OK to preflight request

    try:
        # Parse the input JSON
        data = request.get_json(force=True)

        # Map categorical values to numerical values
        data['Overall Race'] = race_map.get(data['Overall Race'].upper(), -1)  # Default to -1 for unknown races
        data['Day of Week'] = day_of_week_map.get(data['Day of Week'].upper(), -1)  # Default to -1 for unknown days

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

if __name__ == '__main__':
    # Use Heroku-assigned port for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

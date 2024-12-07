from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Allow CORS for all routes (for debugging, later restrict to your domain)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'incident_prediction_model_compressed.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_path}' not found. Ensure it exists in the correct location.")

# Load the city label encoder used during training
city_encoder_path = os.path.join(os.path.dirname(__file__), 'city_label_encoder.pkl')
try:
    city_label_encoder = joblib.load(city_encoder_path)
except FileNotFoundError:
    raise RuntimeError(f"City label encoder file '{city_encoder_path}' not found. Ensure it exists in the correct location.")

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
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)

        # Map and validate inputs (same as before)
        data['Overall Race'] = race_map.get(data['Overall Race'].upper(), -1)
        data['Day of Week'] = day_of_week_map.get(data['Day of Week'].upper(), -1)
        try:
            data['City'] = city_label_encoder.transform([data['City']])[0]
        except ValueError:
            return jsonify({'error': 'Invalid city name'}), 400

        if -1 in (data['Overall Race'], data['Day of Week']):
            return jsonify({'error': 'Invalid categorical input values'}), 400

        input_data = pd.DataFrame({
            'Victim Age': [data['Victim Age']],
            'Overall Race': [data['Overall Race']],
            'City': [data['City']],
            'Hour': [data['Hour']],
            'Day of Week': [data['Day of Week']],
            'Day of Month': [data['Day of Month']],
            'Month': [data['Month']]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        # Return prediction and probability as a decimal
        return jsonify({
            'prediction': prediction,
            'probability': float(probability)  # Already in decimal form
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use Heroku-assigned port for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

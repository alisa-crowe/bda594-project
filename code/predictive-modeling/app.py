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
    """Route to handle prediction requests."""
    if request.method == 'OPTIONS':
        # Respond to preflight request
        return '', 204

    try:
        # Parse the input JSON
        data = request.get_json(force=True)

        # Map categorical values to numerical values
        data['Overall Race'] = race_map.get(data['Overall Race'].upper(), -1)  # Default to -1 for unknown races
        data['Day of Week'] = day_of_week_map.get(data['Day of Week'].upper(), -1)  # Default to -1 for unknown days
        try:
            data['City'] = city_label_encoder.transform([data['City']])[0]
        except ValueError:
            return jsonify({'error': 'Invalid city name'}), 400

        # Validate input
        if -1 in (data['Overall Race'], data['Day of Week']):
            return jsonify({'error': 'Invalid categorical input values'}), 400

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Victim Age': [data['Victim Age']],
            'Overall Race': [data['Overall Race']],
            'City': [data['City']],
            'Hour': [data['Hour']],
            'Day of Week': [data['Day of Week']],
            'Day of Month': [data['Day of Month']],
            'Month': [data['Month']]
        })

        # Predict the outcome
        prediction = model.predict(input_data)

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)

        # Create response with prediction and probabilities
        response = {
            'prediction': prediction[0],  # The predicted class
            'probabilities': {  # Probabilities for each class
                str(idx): prob for idx, prob in enumerate(probabilities[0])
            }
        }

        # Return the response
        return jsonify(response)
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use Heroku-assigned port for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

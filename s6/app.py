from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model from the file
model = joblib.load('salary_prediction_model.pkl')

@app.route('/')
def index():
    # Render the HTML file for the frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (from the frontend form)
    data = request.json

    # Convert the received data into a DataFrame
    # Ensure the data format matches the expected model features
    features = pd.DataFrame([data])

    # Perform one-hot encoding as done during training
    features_encoded = pd.get_dummies(features, drop_first=True)

    # Align the input data with the model's expected feature columns
    model_features = pd.DataFrame(columns=model.feature_names_in_)  # Add missing columns to align with model input
    features_final = pd.concat([features_encoded, model_features], axis=1).fillna(0)  # Fill missing columns with zeros

    # Make the prediction using the loaded model
    prediction = model.predict(features_final)

    # Return the prediction as a JSON response
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

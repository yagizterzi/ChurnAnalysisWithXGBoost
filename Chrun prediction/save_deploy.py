import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from fullproject import model , preprocess_bank_data
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_model_for_deployment(df):
    """Prepare and train model for deployment"""
    # Prepare features and target
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
               'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender'] + \
               [col for col in df.columns if col.startswith('Geography_')]
    
    X = df[features]
    y = df['Exited']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, X_train.columns, scaler


# 1. Save the model and preprocessing components
def save_model(model, feature_names, scaler, output_path='model/'):
    """Save the trained model and necessary preprocessing components"""
    
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model
    joblib.dump(model, output_path + 'xgboost_model.pkl')
    
    # Save feature names
    with open(output_path + 'feature_names.json', 'w') as f:
        json.dump(list(feature_names), f)
    
    # Save the scaler
    joblib.dump(scaler, output_path + 'scaler.pkl')
    
    print(f"Model and components saved to {output_path}")

# 2. Create prediction pipeline
class ChurnPredictor:
    def __init__(self, model_path='model/'):
        # Load model
        self.model = joblib.load(model_path + 'xgboost_model.pkl')
        
        # Load feature names
        with open(model_path + 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load scaler
        self.scaler = joblib.load(model_path + 'scaler.pkl')
    
    def prepare_input(self, data):
        """Prepare input data for prediction"""
        # Convert input to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Return data with correct feature order
        return data[self.feature_names]
    
    def predict(self, data):
        """Make predictions"""
        # Prepare input data
        prepared_data = self.prepare_input(data)
        
        # Make prediction
        prediction = self.model.predict_proba(prepared_data)[:, 1]
        
        return prediction

# 3. Create Flask API
app = Flask(__name__)
predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Make prediction
        prediction = predictor.predict(pd.DataFrame([data]))
        
        # Return prediction
        return jsonify({
            'churn_probability': float(prediction[0]),
            'likely_to_churn': bool(prediction[0] > 0.5)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 4. Save and deploy the model
def deploy_model(model, feature_names, scaler):
    """Save model and start API server"""
    # Save model and components
    save_model(model, feature_names, scaler)
    
    # Initialize predictor
    global predictor
    predictor = ChurnPredictor()
    
    # Start Flask server
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000)

# Example usage


# Modify the main block
if __name__ == "__main__":
    df=pd.read_csv(r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\Chrun prediction\dataset\Churn_Modelling.csv')
    
    df_processed = preprocess_bank_data(df)
    
    # Prepare model and components
    model, feature_names, scaler = prepare_model_for_deployment(df_processed)
    
    # Save model and components
    save_model(model, feature_names, scaler)
    
    # Deploy
    deploy_model(model, feature_names, scaler)
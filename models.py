import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load trained model, scaler, and feature names."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('features.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        raise FileNotFoundError("Run 'python train_model.py' first to train and save model!")

def predict_breast_cancer(features_dict):
    """Predict tumor type from feature dictionary."""
    model, scaler, feature_names = load_model()
    
    # Ensure all features present and in correct order
    input_df = pd.DataFrame([features_dict], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        'prediction': 'Malignant' if prediction == 1 else 'Benign',
        'confidence': max(probability) * 100,
        'prob_malignant': probability[1] * 100,
        'prob_benign': probability[0] * 100
    }


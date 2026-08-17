import os
import joblib

def load_model_artifacts(disease: str):
    \"\"\"
    Loads the saved model and preprocessing pipeline for a given disease.
    
    Args:
        disease (str): Name of the disease ('heart', 'diabetes', 'breast_cancer')
        
    Returns:
        tuple: (model, scaler/preprocessor pipeline)
    \"\"\"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models', disease)
    
    model_path = os.path.join(model_dir, 'best_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

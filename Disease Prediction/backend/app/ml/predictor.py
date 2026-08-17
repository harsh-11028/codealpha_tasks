import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from app.config import settings


class MLPredictor:
    """
    Loads and manages ML models for disease prediction.
    Models are loaded once at startup and reused for all predictions.
    Each model is a complete sklearn Pipeline (preprocessor + classifier).
    """

    def __init__(self):
        self.pipelines: Dict[str, Any] = {}       # Complete sklearn Pipeline
        self.metadata: Dict[str, Dict] = {}        # Model metadata (features, algorithm, etc.)
        self.supported_diseases = ['heart', 'diabetes', 'breast_cancer']
        self.is_loaded = False

    def load_models(self) -> None:
        """Load all disease ML pipelines from disk at startup."""
        base_path = settings.ML_MODELS_PATH
        print(f"\n[MLPredictor] Loading models from: {base_path}")

        for disease in self.supported_diseases:
            model_path = os.path.join(base_path, disease, "best_model.joblib")
            metadata_path = os.path.join(base_path, disease, "metadata.json")

            try:
                if not os.path.exists(model_path):
                    print(f"  [WARNING] Model not found for '{disease}' at: {model_path}")
                    print(f"  [INFO] Run: python ml/train_{disease}.py to train the model first.")
                    continue

                # Load complete pipeline (preprocessor + classifier)
                self.pipelines[disease] = joblib.load(model_path)

                # Load metadata
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata[disease] = json.load(f)
                else:
                    self.metadata[disease] = {'algorithm': 'Unknown', 'features': []}

                algo = self.metadata[disease].get('algorithm', 'Unknown')
                acc = self.metadata[disease].get('accuracy', 0)
                print(f"  ✓ Loaded '{disease}' model: {algo} (Accuracy: {acc:.4f})")

            except Exception as e:
                print(f"  ✗ Error loading model for '{disease}': {str(e)}")

        loaded_count = len(self.pipelines)
        total = len(self.supported_diseases)
        self.is_loaded = loaded_count > 0
        print(f"\n[MLPredictor] {loaded_count}/{total} disease models loaded.\n")

    def predict(self, disease: str, input_data: dict) -> Tuple[int, float, str]:
        """
        Make a prediction for a given disease using the loaded pipeline.

        Args:
            disease: One of 'heart', 'diabetes', 'breast_cancer'
            input_data: Dict of feature name → value (must match trained features)

        Returns:
            Tuple of (prediction: int, probability: float, model_name: str)
        """
        if disease not in self.supported_diseases:
            raise ValueError(f"Unsupported disease: '{disease}'. Must be one of {self.supported_diseases}")

        if disease not in self.pipelines:
            raise RuntimeError(
                f"Model for '{disease}' is not loaded. "
                f"Please run 'python ml/train_{disease}.py' to train the model first."
            )

        pipeline = self.pipelines[disease]
        meta = self.metadata.get(disease, {})
        expected_features = meta.get('features', [])

        # Validate input features
        if expected_features:
            missing = [f for f in expected_features if f not in input_data]
            if missing:
                raise ValueError(f"Missing features for '{disease}' prediction: {missing}")

            # Build DataFrame with correct column order
            df = pd.DataFrame([{k: input_data[k] for k in expected_features}])
        else:
            # Fallback: use all input data as-is
            df = pd.DataFrame([input_data])

        # The pipeline handles both preprocessing AND prediction
        # No need for separate scaling step
        prediction = int(pipeline.predict(df)[0])

        if hasattr(pipeline, "predict_proba"):
            proba_all = pipeline.predict_proba(df)[0]
            # probability of positive class (class=1)
            probability = float(proba_all[1])
        else:
            probability = float(prediction)

        # Get classifier name from pipeline
        if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
            model_name = type(pipeline.named_steps['classifier']).__name__
        else:
            model_name = meta.get('algorithm', type(pipeline).__name__)

        return prediction, probability, model_name

    def get_model_status(self) -> Dict:
        """Return status of all loaded models."""
        return {
            disease: {
                'loaded': disease in self.pipelines,
                'algorithm': self.metadata.get(disease, {}).get('algorithm', 'Not trained'),
                'accuracy': self.metadata.get(disease, {}).get('accuracy', None),
                'trained_at': self.metadata.get(disease, {}).get('trained_at', None),
            }
            for disease in self.supported_diseases
        }

    def get_features(self, disease: str) -> list:
        """Return the expected feature names for a disease."""
        return self.metadata.get(disease, {}).get('features', [])


# Singleton instance
predictor = MLPredictor()

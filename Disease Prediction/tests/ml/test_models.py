"""
Tests for ML model loading and prediction functionality.
Run with: pytest tests/ml/ -v
"""
import pytest
import os
import json
import numpy as np
import sys

# Add ml directory to path
ML_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ml')
sys.path.insert(0, ML_DIR)


class TestModelArtifacts:
    """Test that all model artifacts exist and are loadable."""

    DISEASES = ['heart', 'diabetes', 'breast_cancer']

    def test_model_files_exist(self):
        """Test that all model artifact files exist."""
        for disease in self.DISEASES:
            model_dir = os.path.join(ML_DIR, 'models', disease)
            assert os.path.exists(os.path.join(model_dir, 'best_model.joblib')), \
                f"Missing best_model.joblib for {disease}"
            assert os.path.exists(os.path.join(model_dir, 'metadata.json')), \
                f"Missing metadata.json for {disease}"
            assert os.path.exists(os.path.join(model_dir, 'all_metrics.json')), \
                f"Missing all_metrics.json for {disease}"

    def test_metadata_structure(self):
        """Test that metadata.json has correct structure."""
        required_keys = ['disease', 'algorithm', 'features', 'accuracy',
                         'precision', 'recall', 'f1_score', 'roc_auc', 'trained_at']
        for disease in self.DISEASES:
            metadata_path = os.path.join(ML_DIR, 'models', disease, 'metadata.json')
            with open(metadata_path) as f:
                metadata = json.load(f)
            for key in required_keys:
                assert key in metadata, f"Missing key '{key}' in {disease} metadata"
            assert isinstance(metadata['features'], list)
            assert len(metadata['features']) > 0

    def test_model_accuracy_reasonable(self):
        """Test that model accuracy is within a reasonable range (not random)."""
        for disease in self.DISEASES:
            metadata_path = os.path.join(ML_DIR, 'models', disease, 'metadata.json')
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Accuracy should be > 50% (better than random)
            assert metadata['accuracy'] > 0.5, \
                f"{disease} model accuracy {metadata['accuracy']} is too low"

    def test_all_metrics_structure(self):
        """Test that all_metrics.json contains metrics for all 4 algorithms."""
        expected_algorithms = ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'XGBClassifier']
        for disease in self.DISEASES:
            metrics_path = os.path.join(ML_DIR, 'models', disease, 'all_metrics.json')
            with open(metrics_path) as f:
                all_metrics = json.load(f)
            for algo in expected_algorithms:
                assert algo in all_metrics, f"Missing metrics for {algo} in {disease}"


class TestModelLoading:
    """Test loading models with joblib."""

    def test_load_heart_model(self):
        import joblib
        model_path = os.path.join(ML_DIR, 'models', 'heart', 'best_model.joblib')
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_load_diabetes_model(self):
        import joblib
        model_path = os.path.join(ML_DIR, 'models', 'diabetes', 'best_model.joblib')
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')

    def test_load_breast_cancer_model(self):
        import joblib
        model_path = os.path.join(ML_DIR, 'models', 'breast_cancer', 'best_model.joblib')
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')


class TestModelPredictions:
    """Test that models produce valid predictions."""

    def _load_model(self, disease):
        import joblib
        model_path = os.path.join(ML_DIR, 'models', disease, 'best_model.joblib')
        return joblib.load(model_path)

    def _load_metadata(self, disease):
        metadata_path = os.path.join(ML_DIR, 'models', disease, 'metadata.json')
        with open(metadata_path) as f:
            return json.load(f)

    def test_heart_prediction_output(self):
        """Test heart model prediction output."""
        import pandas as pd
        model = self._load_model('heart')
        meta = self._load_metadata('heart')

        sample = {
            'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 160, 'exang': 0,
            'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
        }
        df = pd.DataFrame([{k: sample[k] for k in meta['features']}])
        pred = model.predict(df)
        prob = model.predict_proba(df)

        assert pred[0] in [0, 1]
        assert prob.shape[1] == 2
        assert 0.0 <= prob[0][1] <= 1.0

    def test_diabetes_prediction_output(self):
        """Test diabetes model prediction output."""
        import pandas as pd
        model = self._load_model('diabetes')
        meta = self._load_metadata('diabetes')

        sample = {
            'Pregnancies': 2, 'Glucose': 120, 'BloodPressure': 70,
            'SkinThickness': 20, 'Insulin': 80, 'BMI': 25.5,
            'DiabetesPedigreeFunction': 0.5, 'Age': 35
        }
        df = pd.DataFrame([{k: sample[k] for k in meta['features']}])
        pred = model.predict(df)
        prob = model.predict_proba(df)

        assert pred[0] in [0, 1]
        assert 0.0 <= prob[0][1] <= 1.0

    def test_breast_cancer_prediction_output(self):
        """Test breast cancer model prediction output."""
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        model = self._load_model('breast_cancer')
        meta = self._load_metadata('breast_cancer')

        # Use first sample from sklearn dataset
        data = load_breast_cancer()
        sample_values = data.data[0]
        feature_names = meta['features']  # underscore version
        sample = {feat: float(val) for feat, val in zip(feature_names, sample_values)}

        df = pd.DataFrame([sample])
        pred = model.predict(df)
        prob = model.predict_proba(df)

        assert pred[0] in [0, 1]
        assert 0.0 <= prob[0][1] <= 1.0

    def test_prediction_batch(self):
        """Test model predictions on a batch of inputs."""
        import pandas as pd
        import joblib
        model = self._load_model('diabetes')
        meta = self._load_metadata('diabetes')

        samples = [
            {'Pregnancies': i, 'Glucose': 100 + i*10, 'BloodPressure': 70,
             'SkinThickness': 20, 'Insulin': 80, 'BMI': 25.0 + i,
             'DiabetesPedigreeFunction': 0.3 + i*0.1, 'Age': 30 + i}
            for i in range(5)
        ]
        df = pd.DataFrame([{k: s[k] for k in meta['features']} for s in samples])
        preds = model.predict(df)
        probs = model.predict_proba(df)

        assert len(preds) == 5
        assert all(p in [0, 1] for p in preds)
        assert probs.shape == (5, 2)

    def test_feature_order_matters(self):
        """Test that correct feature order is preserved."""
        import pandas as pd
        model = self._load_model('heart')
        meta = self._load_metadata('heart')

        assert meta['features'] == ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


class TestPreprocessingPipeline:
    """Test that preprocessing handles edge cases correctly."""

    def test_pipeline_handles_missing_values(self):
        """Test that pipeline handles NaN values via SimpleImputer."""
        import pandas as pd
        import joblib
        import numpy as np

        model = self._load_model()
        meta = self._load_metadata()

        sample = {
            'Pregnancies': 2, 'Glucose': np.nan, 'BloodPressure': 70,
            'SkinThickness': np.nan, 'Insulin': np.nan, 'BMI': 25.5,
            'DiabetesPedigreeFunction': 0.5, 'Age': 35
        }
        df = pd.DataFrame([{k: sample[k] for k in meta['features']}])
        # Should not raise exception (SimpleImputer handles NaN)
        pred = model.predict(df)
        assert pred[0] in [0, 1]

    def _load_model(self):
        import joblib
        return joblib.load(os.path.join(ML_DIR, 'models', 'diabetes', 'best_model.joblib'))

    def _load_metadata(self):
        with open(os.path.join(ML_DIR, 'models', 'diabetes', 'metadata.json')) as f:
            return json.load(f)
